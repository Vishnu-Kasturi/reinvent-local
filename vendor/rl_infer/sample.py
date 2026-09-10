#SMILES Generative model - RNN encoder-decoder architecture - PyTorch implementation

import sys
import csv
import os
import pandas as pd
import numpy as np
import pickle
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss
import time
import networkx as nx
import scipy.sparse as sp
from scipy.special import expit

from torch import optim
from torch.utils import data
from torch.autograd import Variable
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from rdkit import Chem
from io import open

from rdkit import rdBase
rdBase.DisableLog('rdApp.error') #Suppresses error messages from rdkit when parsing SMILES strings to RDKit molecules

os.environ['CUDA_VISIBLE_DEVICES'] = '0' #Force keras to use CPU for calculations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #PyTorch will figure out which device to use for training
#device = torch.device("cpu")
print("Device available:",device)
#print("Current GPU in use:",torch.cuda.current_device())
#print("Name of GPU in use:",torch.cuda.get_device_name(torch.cuda.current_device()))
cpu = torch.device("cpu")


#Command-line arguments - File names
protein_index = sys.argv[1]
char_to_int_file = sys.argv[2]
int_to_char_file = sys.argv[3]
graphpath = sys.argv[4]
combined_cptfile = sys.argv[5]
outfile = sys.argv[6]

#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------GRAPH VAE FUNCTION DEFINITIONS------------------------------------------------
#Function to load the dataset and output adjacency matrix and feature matrix from the data
def load_data(receptor, path):
	#'x' is the node feature one-hot encoding (scipy sparse CSR matrix) and 'graph' is the receptor binding site graph (collections defaultdict)
	names = ['x', 'graph']
	objects = []
	for i in range(len(names)):
		objects.append(pickle.load(open(str(path)+"{}.{}".format(receptor, names[i]), 'rb')))
	x, graph = tuple(objects)
	features = x.tolil()
	features = torch.FloatTensor(np.array(features.todense())) #Converting the features to torch tensor because no preprocessing is required for feature vector
	adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))  #In-built networkx functions to convert a dict of lists to graph and then into an adjacency matrix
	#print(receptor, features.size())
	return adj, features

#Function to convert scipy sparse matrices to tuples for faster computations
def sparse_to_tuple(sparse_matrix):
	if not sp.isspmatrix_coo(sparse_matrix):
		sparse_matrix = sparse_matrix.tocoo()
	coords = np.vstack((sparse_matrix.row, sparse_matrix.col)).transpose()
	values = sparse_matrix.data
	shape = sparse_matrix.shape
	return coords, values, shape

#Convert a scipy sparse matrix to a torch sparse tensor - For passing the inputs into the PyTorch models
def sparse_matrix_to_torch_sparse_tensor(sparse_matrix):
	sparse_matrix = sparse_matrix.tocoo().astype(np.float32)
	indices = torch.from_numpy(np.vstack((sparse_matrix.row, sparse_matrix.col)).astype(np.int64))
	values = torch.from_numpy(sparse_matrix.data)
	shape = torch.Size(sparse_matrix.shape)
	return torch.sparse.FloatTensor(indices, values, shape)

#Function to preprocess the adjacency matrix for applying the propagation rule of spectral graph convolutions (Kipf and Welling, 2016)
def preprocess_graph(adj):
	adj = sp.coo_matrix(adj)
	adj_hat = adj + sp.eye(adj.shape[0]) #Adding self-loops to the graph by sum of adjacency matrix and a identity matrix (sp.eye) of same shape - This enables inclusion of the node features itself in the graph embedding rather than only its neighborhood
	rowsum = np.array(adj_hat.sum(1)) #Calculating the degree matrix of the graph
	degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())  #Calculating the diagonalized degree matrix to the power -0.5
	adj_normalized = adj_hat.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()  #D^-0.5*adj_hat*D^-0.5 - Normalizing the adjacency matrix - To not allow the scale of the features to change on multiplication with adjacency matrix
	# return sparse_to_tuple(adj_normalized)
	adj_normalized = sparse_matrix_to_torch_sparse_tensor(adj_normalized)
	adj_normalized = adj_normalized.to_dense() #To avoid 'RuntimeError: sparse tensors do not have storage' - Should be a DENSE tensor (Error when using PyTorch DataLoader to load, preprocess and shuffle batches)
	return adj_normalized
#---------------------------------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------SMILES VAE MAJOR FUNCTION DEFINITIONS------------------------------------------------
#Take a SMILES string and return it in '*' tokenized form - To account for bisyllable atoms
def tokenize(smiles):
	bisyllable_atoms = ['He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'Cl', 'Ar', 'Ca', 'Ti', 'Cr', 'Fe', 'Ni', 'Cu', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Sb', 'Te', 'Xe', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Re', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'At', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'Lr', 'Rf', 'Db', 'Sg', 'Mt', 'Ds', 'Rg', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Zn', 'Mn'] #List of atoms with two letter representations in the periodic table

	smiles.strip()
	tokenized="" #An empty string to append the tokenized string
	myiter = iter(range(1,len(smiles)))
	for i in myiter:  #Index starts at 1 to prevent array index out of bounds error
		#print(i)
		test=smiles[i-1]+smiles[i]  #Define a 2 character window
		#print(smiles[i-1]+","+smiles[i])
		if test in bisyllable_atoms:  #Check if the 2 characters form a meaningful atom type
			tokenized=tokenized+test+"*" #If so put the token after the atom type
			if i<len(smiles)-1: #Prevents from generating StopIteration exception
				next(myiter) #Skip the next iteration if a bisyllable-atom occurs
		elif test in (atom.lower() for atom in bisyllable_atoms): #If the atom occurs as part of an aromatic ring like [te+]
			tokenized=tokenized+test+"*"
			if i<len(smiles)-1: #Prevents from generating StopIteration exception
				next(myiter) #Skip the next iteration if a bisyllable-atom occurs
		else:
			tokenized=tokenized+smiles[i-1]+"*"
			if i==len(smiles)-1:
				tokenized=tokenized+smiles[i]+"*"

	tokenized = tokenized[:-1]
	return tokenized


#Given a SMILES dataset tokenize, find the unique characters and prepare a two-way dictionary - Return all outputs
#def prepare_dicts_and_dictfiles(smiles, char_to_int_file, int_to_char_file):
def prepare_dicts_and_dictfiles(char_to_int_file, int_to_char_file):	
	with open(char_to_int_file, 'rb') as handle:
		char_to_int = pickle.loads(handle.read())

	with open(int_to_char_file, 'rb') as handle:
		int_to_char = pickle.loads(handle.read())

	charset = char_to_int.keys()

	return charset, char_to_int, int_to_char


#Given a tokenized SMILES dataset split it into X and Y one-hot encoded numpy tensors and return them
def prepare_dataset(smiles):
	one_hot =  np.zeros((smiles.shape[0], embed), dtype=np.int64)
	for i,smile in enumerate(smiles):
		#encode the startchar
		one_hot[i,0] = char_to_int["!"]
		#encode the rest of the chars
		molchar=smile.split("*") #Split the SMILES based on the token
		for j,c in enumerate(molchar): #Pass the split atoms and characters to the one-hot encoder
			one_hot[i,j+1] = char_to_int[c]
		#Encode endchar
		one_hot[i,len(molchar)+1:] = char_to_int["E"] #The length of the SMILES=len(molchar)

	#Return two, one for input and the other for output
	return one_hot[:,0:-1], one_hot[:,1:] #Input has -1 to indicate all columns from 0 except the last column, Output has all columns except the ! mark in the first column
#--------------------------------------------------------------------------------------------------------------------------------------



#---------------------------------------------------PYTORCH DATASET CLASS DEFINITION---------------------------------------------------
class Dataset(data.Dataset):
	def __init__(self, graph_dataset, smiles_dataset, path):
		self.data = graph_dataset
		self.path = path
		self.adj = []
		self.adj_labels = []
		self.node_features = []
		self.node_features_labels = []
		self.pos_weights = []
		self.loss_norms = []

		self.smiles_subset = []

		#All pre-processing of the data before creating mini-batches is done within the class
		for i in self.data:
			A, X = load_data(i, self.path)
			A_norm = preprocess_graph(A)
			#A_pred = A + A.T  #T represents transpose of the matrix - If matrix does not include edges in both directions
			A_pred = A  #Need not add the transpose of A because the matrix already includes edges in both directions
			A_labels = A_pred + sp.eye(A_pred.shape[0])  #For calculating the loss function the adjacency matrix must have self-loops
			A_labels = torch.FloatTensor(A_labels.toarray())

			X_labels = X  #TODO: When the code for predicting X is added, pre-processing of X to X_labels should be included

			#Parameters for loss function - Depends on the adjacency matrix values before all pre-processing is done
			#Only applicable for UNWEIGHTED adjacency matrix - pos_weight should be a tensor of values for weighted adj matrices
			pos_weight = float(A.shape[0] * A.shape[0] - A.sum()) / A.sum()  #A SCALAR tensor for BINARY classification
			norm = A.shape[0] * A.shape[0] / float((A.shape[0] * A.shape[0] - A.sum()) * 2)	

			self.adj.append(A_norm)
			self.adj_labels.append(A_labels)
			self.node_features.append(X)
			self.node_features_labels.append(X_labels)
			self.pos_weights.append(pos_weight)
			self.loss_norms.append(norm)

			smiles = smiles_dataset[i]
			self.smiles_subset.append(smiles)
			#print(i)

		self.smiles_subset = np.array(self.smiles_subset)
		self.smiles_X, self.smiles_Y = prepare_dataset(self.smiles_subset)

	def __len__(self):
		return len(self.data)

	#Given an index in a mini-batch, returns the corresponding X and Y for that index
	def __getitem__(self, index):
		#print(index)
		X_out = torch.from_numpy(self.smiles_X[index,:])
		Y_out = torch.from_numpy(self.smiles_Y[index,:])
		X_adj = self.adj[index]
		X_features = self.node_features[index]
		Y_adj = self.adj_labels[index]
		Y_features = self.node_features_labels[index]
		pos_weight = self.pos_weights[index]
		norm = self.loss_norms[index]
		#print(index, X_out)
		return X_out, Y_out, X_adj, X_features, Y_adj, Y_features, pos_weight, norm
#----------------------------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------GAT_VAE_LAYERS---------------------------------------------------------
#--------------------------------------------------GRAPH ATTENTION LAYER DEFINITION----------------------------------------------
class GraphAttentionLayer(nn.Module):
	"""
	Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
	"""

	def __init__(self, input_size, output_size, dropout=0.2, activation=nn.LeakyReLU, alpha=0.2, training=True, concat=True):
		super(GraphAttentionLayer, self).__init__()
		self.dropout = dropout
		self.in_features = input_size
		self.out_features = output_size
		self.alpha = alpha
		self.concat = concat
		self.activation = activation
		self.training = training

		self.W = nn.Parameter(torch.zeros(size=(self.in_features, self.out_features)))  #Linear transformation, W(FxF') applied on every node
		torch.nn.init.xavier_uniform_(self.W)  #Glorot initialization
		self.a = nn.Parameter(torch.zeros(size=(2*self.out_features, 1)))  #Attention mechanism, a(2F'x1) applied on all pairs of nodes
		torch.nn.init.xavier_uniform_(self.a)  #Glorot initialization

		self.leakyrelu = self.activation(self.alpha)  #Activation applied before Softmax over the attention coefficients, eij

	def forward(self, input, adj):
		#print(input.size(), adj.size(), self.W.size(), self.a.size()) #torch.Size([51, 29]) torch.Size([51, 51]) torch.Size([29, 512]) torch.Size([1024, 1])
		h = torch.mm(input, self.W)  #Input is the node feature vector (NxF mm FxF' = NxF') = Whj matrix in the paper - torch.Size([51, 512])
		N = h.size()[0]  #N is the number of nodes in the input graph - 51

		a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)  #[N, N, 2F']
		#print(h.repeat(1, N).size(), h.repeat(N, 1).size())
		#[N*N, F'] [N*N, F'] - Concatenation gives [N*N, 2F'] - Viewed as [N, N, 2F'] -> (Whi||Whj) step - [51, 51, 1024]
 
		e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  #Computing the attention coefficients for all node pairs (irrespective of whether an edge is present between them or not) - a(Whi||Whj) step  #[N, N, 1].squeeze(2) = [N, N] = [51, 51]

		zero_vec = -9e15*torch.ones_like(e)  #Creating a vector of 1s to the dimension of eij
		attention = torch.where(adj > 0, e, zero_vec)  #Considering the neighborhood of every node (where adj > 0, there is an edge present)
		attention = F.softmax(attention, dim=1)  #Normalizing the attention coefficients - Activation has already been applied
		attention = F.dropout(attention, self.dropout, training=self.training)
		h_prime = torch.matmul(attention, h)  #Output features (hj' vector in the paper) - [N, F'] = [51, 512]

		#If there are multiple parallel attention layers and we want to concatenate those parallel attention coefficients, a non-linearity must be applied before concatenation as per the paper. Otherwise, we can directly use the coefficients without non-linearity.
		if self.concat:
			return F.elu(h_prime)
		else:
			return h_prime

	def __repr__(self):
		return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


#--------------------------------------------------GRAPH VAE ENCODER NETWORK DEFINITION--------------------------------------------------
class GAT_VAE_Encoder(nn.Module):
	def __init__(self, inp_feature_dim, hidden_size, z_dim, dropout=0.2, training=True):
	#def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
		"""Dense version of GAT."""
		super(GAT_VAE_Encoder, self).__init__()
		self.inp_feature_dim = inp_feature_dim
		self.hidden_size = hidden_size
		self.z_dim = z_dim
		self.dropout = dropout
		self.training = training

		self.attentions1 = [GraphAttentionLayer(inp_feature_dim, hidden_size, dropout=dropout, training=self.training, concat=False) for _ in range(5)]  #nheads replaced by a number - If number > 1, set concat=True (Multiple parallel attention heads)
		for i, attention in enumerate(self.attentions1):
			self.add_module('attention_{}'.format(i), attention)

		self.mean = GraphAttentionLayer(hidden_size, z_dim, dropout=dropout, training=self.training)
		self.logvar = GraphAttentionLayer(hidden_size, z_dim, dropout=dropout, training=self.training)
		self.out_att = GraphAttentionLayer(5 * hidden_size, hidden_size, dropout=dropout, training=self.training, concat=False)  #Multi-head attention

	def forward(self, x, adj):
		x = F.dropout(x[0], self.dropout, training=self.training)
		x = torch.cat([att(x, adj[0]) for att in self.attentions1], dim=1)
		x = F.dropout(x, self.dropout, training=self.training)
		x = self.out_att(x, adj[0])  #torch.Size([46, 512]) - [N, hidden_size]
		x = F.elu(x)
		mu = self.mean(x, adj[0])  #[N, z_dim]
		sigma = self.logvar(x, adj[0])  #[N, z_dim]
		return mu, sigma


#----------------------------------------REPARAMETERIZATION TRICK - LATENT VECTOR GENERATION----------------------------------------
#Used to enable backpropagation through a sampling process - Approximating the latent manifold using a mixture of Gaussians
def reparameterization_trick_graph(mu, sigma):  #TODO: Check if mu alone must be returned because, the model has been set to eval()
	std = torch.exp(sigma)
	eps = torch.randn_like(std)
	latent_z = eps.mul(std).add_(mu)
	return latent_z

#----------------------------------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------SMILES_VAE_LAYERS---------------------------------------------------------
#----------------------------------------------STACK-AUGMENTED GRU LAYER DEFINITION------------------------------------------------
class Stack_GRU(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, stack_depth, stack_width, batch_size, dropout=0.2, n_layers=1, bidir=False):
		super(Stack_GRU, self).__init__()

		#Input parameter definitions
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.stack_depth = stack_depth
		self.stack_width = stack_width
		self.num_layers = n_layers
		self.bidir = bidir
		self.batch_size = batch_size

		#Layer definitions
		self.embedding = nn.Embedding(input_size, hidden_size)
		self.stack_controls_layer = nn.Linear(in_features=hidden_size, out_features=3)
		self.stack_input_layer = nn.Linear(in_features=hidden_size, out_features=stack_width)
		self.gru = nn.GRU(input_size=hidden_size+stack_width, hidden_size=hidden_size, num_layers=n_layers, bidirectional=bidir, dropout=dropout)

	#Forward function of the neural network layer
	def forward(self, inp, hidden, stack):
		embedded_input = self.embedding(inp.view(1,-1))  #[1,batch_size,hidden_size]
		hidden_ = hidden  #[num_layers*num_dir, batch_size, hidden_size]
		hidden_2_stack = hidden_[-1, :, :]  #Take only the last unidirectional layer hidden state
		stack_controls = self.stack_controls_layer(hidden_2_stack)  #[hidden_size, 3]
		stack_controls = F.softmax(stack_controls, dim=-1)  #[hidden_size, 3]
		stack_input = self.stack_input_layer(hidden_2_stack.unsqueeze(0))  #[batch_size, hidden_size*numdir]
		stack_input = torch.tanh(stack_input)  #[1, batch_size, stack_width]
		stack = self.stack_augmentation(stack_input.permute(1, 0, 2), stack, stack_controls)  #[batch_size, stack_depth, stack_width]
		stack_top = stack[:, 0, :].unsqueeze(0)
		inp = torch.cat((embedded_input, stack_top), dim=2)
		output, hidden = self.gru(inp, hidden)
		return output, hidden, stack

	#Function to calculate the probabilities of the 3 stack operations
	def stack_augmentation(self, input_val, prev_stack, controls):
		batch_size = prev_stack.size(0)
		controls = controls.view(-1, 3, 1, 1)  #[1,batch_size,3] --> [batch_size,3,1,1]
		zeros_at_the_bottom = torch.zeros(batch_size, 1, self.stack_width, device=device)
		a_push, a_pop, a_no_op = (controls[:, 0], controls[:, 1], controls[:, 2])  #[batch_size,1,1]
		stack_down = torch.cat((prev_stack[:, 1:], zeros_at_the_bottom), dim=1)  #[batch_size,stack_depth,stack_width] - POP loses first element and fills the bottom of stack with zeros
		stack_up = torch.cat((input_val, prev_stack[:, :-1]), dim=1)  #[batch_size,stack_depth,stack_width] - PUSH loses last element by concatenating the stack without last element to the bottom of input_val
		new_stack = a_no_op * prev_stack + a_push * stack_up + a_pop * stack_down
		return new_stack	

	#Function to initialize hidden state of the GRU cell
	def initHidden(self):
		return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=device)

	#Function to initialize the stack
	def initStack(self):
		return torch.zeros(batch_size, self.stack_depth, self.stack_width, device=device)


#----------------------------------------------------VAE ENCODER DEFINITION--------------------------------------------------------
class VAE_Encoder(nn.Module):
	def __init__(self, input_size, hidden_size, z_dim, output_size, batch_size, stack_depth, stack_width, dropout=0.2, n_layers=1, bidir=False):
		super(VAE_Encoder, self).__init__()

		self.hidden_size = hidden_size
		self.num_layers = n_layers
		self.stack_depth = stack_depth
		self.stack_width = stack_width
		self.bidir = bidir
		self.batch_size = batch_size
		self.numdir=1

		self.forward_stackgru = Stack_GRU(input_size, hidden_size, output_size, stack_depth, stack_width, batch_size, dropout=dropout, n_layers=n_layers, bidir=False)

		if(self.bidir):
			self.numdir=2
			self.backward_stackgru = Stack_GRU(input_size, hidden_size, output_size, stack_depth, stack_width, batch_size, dropout=dropout, n_layers=n_layers, bidir=False)

		self.mean = nn.Linear(self.numdir*hidden_size, z_dim)
		self.logvar = nn.Linear(self.numdir*hidden_size, z_dim)


	#Forward function for unidirectional Stack-GRU layers in the encoder
	def forward_unidir(self, inp, hidden_forward, stack_forward):
		output, hidden_forward, stack_forward = self.forward_stackgru(inp, hidden_forward, stack_forward)  #Calls the forward function of Stack_GRU
		#print(hidden_forward.size())
		return hidden_forward


	#Forward function for bidirectional Stack-GRU layers in the encoder
	def forward_bidir(self, inp1, inp2, hidden_forward, stack_forward, hidden_backward, stack_backward):
		#Functionality of the forward StackGRU
		output, hidden_forward, stack_forward = self.forward_stackgru(inp1, hidden_forward, stack_forward)
		#Functionality of the backward StackGRU
		output_backward, hidden_backward, stack_backward = self.backward_stackgru(inp2, hidden_backward, stack_backward)
		return hidden_forward, hidden_backward

	def post_gru_reshape_function_unidir(self, hidden):
		hidden = hidden.view(self.num_layers, self.batch_size, self.hidden_size) #[n_layers, batch_size, hidden_size]
		hidden_new = hidden[-1, :, :]  #Only the last layer is considered for mu and sigma calculation
		mu = self.mean(hidden_new)  #[1, batch_size, z_dim]
		sigma = self.logvar(hidden_new)  #[1, batch_size, z_dim]
		return mu, sigma

	def post_gru_reshape_function_bidir(self, hidden1, hidden2):
		hidden1 = hidden1.view(self.num_layers, self.batch_size, self.hidden_size) #[n_layers, batch_size, hidden_size]
		hidden_new_1 = hidden1[-1, :, :]
		hidden2 = hidden2.view(self.num_layers, self.batch_size, self.hidden_size)
		hidden_new_2 = hidden2[-1, :, :]
		#Concatenate the forward and backward hidden states of the last GRU layer of the encoder
		hidden_new = torch.cat([hidden_new_1, hidden_new_2], dim=1)
		mu = self.mean(hidden_new)  #[1, batch_size, z_dim]
		sigma = self.logvar(hidden_new)  #[1, batch_size, z_dim]		
		return mu, sigma

	def load_model_weights(self, path):
		weights = torch.load(path)
		self.load_state_dict(weights) #state_dict is a PyTorch dictionary from layers to parameter tensors

	def save_model_weights(self, path):
		torch.save(self.state_dict(), path)


#---------------------------------------------------VAE DECODER DEFINITION--------------------------------------------------------
class VAE_Decoder(nn.Module):
	def __init__(self, input_size, hidden_size, z_dim, output_size, batch_size, stack_depth, stack_width, dropout=0.2, n_layers=1, bidir=False):
		super(VAE_Decoder, self).__init__()

		self.hidden_size = hidden_size
		self.batch_size = batch_size
		self.output_size = output_size
		self.num_layers = n_layers
		self.stack_depth = stack_depth
		self.stack_width = stack_width
		self.bidir = bidir
		self.numdir=1

		self.latent_to_hidden = nn.Linear(z_dim, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)
		self.decoder_stackgru = Stack_GRU(input_size, hidden_size, output_size, stack_depth, stack_width, batch_size, dropout=dropout, n_layers=n_layers, bidir=False)  #By default the decoder is always unidirectional - Bidirectional decoder has not yet been implemented

	def forward(self, inp, hidden, stack):
		#outputs = torch.zeros(input_length, self.batch_size, self.output_size).to(device)  #[input_length, batch_size, vocab_size]
		output, hidden, stack = self.decoder_stackgru(inp, hidden, stack)
		output = self.out(output)  #[1, batch_size, vocab_size]
		return output, hidden, stack

	def latent_vector_to_hidden(self, latent_z):
		latent_z = latent_z.repeat(self.num_layers, 1, 1) #To get the appropriate hidden layer dimensions [num_layers*num_dir, batch_size, z_dim]
		hidden = self.latent_to_hidden(latent_z)  #[num_layers*num_dir, batch_size, hidden_size]
		return hidden

	def load_model_weights(self, path):
		weights = torch.load(path)
		self.load_state_dict(weights) #state_dict is a PyTorch dictionary from layers to parameter tensors

	def save_model_weights(self, path):
		torch.save(self.state_dict(), path)


#----------------------------------------REPARAMETERIZATION TRICK - LATENT VECTOR GENERATION----------------------------------------
#Used to enable backpropagation through a sampling process - Approximating the latent manifold using a mixture of Gaussians
def reparameterization_trick(mu, sigma):
	std = torch.exp(0.5 * sigma)
	eps = torch.randn_like(std) #Random Gaussian noise epsilon used to approximate the latent random variable, z
	latent_z = eps.mul(std).add_(mu)
	return latent_z


#------------------------------------------KULLBACK-LEIBLER DIVERGENCE WEIGHTING FUNCTION-------------------------------------------
#From IBM PaccMann_RL code (paccmann_chemistry -> utils)
def kl_weight(step, growth_rate=0.004):
	"""Kullback-Leibler weighting function.

	KL divergence weighting for better training of
	encoder and decoder of the VAE.

	Reference:
		https://arxiv.org/abs/1511.06349

	Args:
		step (int): The training step.
		growth_rate (float): The rate at which the weight grows.
			Defaults to 0.0015 resulting in a weight of 1 around step=9000.

	Returns:
		float: The weight of KL divergence loss term.
	"""
	weight = 1 / (1 + math.exp((15 - growth_rate * step)))
	return weight


#---------------------------------------------KULLBACK-LEIBLER DIVERGENCE LOSS FUNCTION-------------------------------------------
def kl_divergence_loss(mu, logvar, kl_growth=0.0015, step=None, eval_mode=False):
	"""KL Divergence loss from VAE paper.

	Reference:
		Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014

	Args:
		mu (torch.Tensor): Encoder output of means of shape
			`[batch_size, input_size]`.
		logvar (torch.Tensor): Encoder output of logvariances of shape
			`[batch_size, input_size]`.
	Returns:
		The KL Divergence of the thus specified distribution and a unit
		Gaussian.
	"""
	# Increase precision (numerical underflow caused negative KLD). - Typecasting explicitly to double causes error in backpropagation?!
	mu = mu.to(device)
	logvar = logvar.to(device)
	kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	#if eval_mode:
		#kl_w = 1.0
	#else:
		#kl_w = kl_weight(step, growth_rate=kl_growth)
		#print(step, kl_w)
	#kld = kld * kl_w
	return kld


#--------------------------------------------------------------------------------------------------------------------------------------
#For the GRAPH + SMILES model
def extract_latent_vector_v1(smiles_encoder, gat_encoder, X_adj, X_feat, prefix):
	#Preparing the graph latent vector
	X_adj = X_adj.to(device)
	X_feat = X_feat.to(device)
	graph_mean, graph_logvar = gat_encoder(X_feat, X_adj)  #Both tensors are [No. of nodes, 256] dimension
	graph_latent_z = reparameterization_trick_graph(graph_mean, graph_logvar)  #[No. of nodes, 256]
	graph_latent_z = torch.mean(graph_latent_z, dim=0)  #[256] - Will average over the node embeddings to get a pooled vector
	graph_latent_z = graph_latent_z.unsqueeze(0)  #[1, 256] - To add a batch_size dimension
	graph_latent_z = graph_latent_z.unsqueeze(0)  #[1, 1, 256]

	#Preparing the SMILES latent vector - with just the "!" character as prefix
	input_tensor = prefix  #Prefix is already a torch tensor with "!" character one-hot encoding
	hidden_forward = smiles_encoder.forward_stackgru.initHidden()
	stack_forward = smiles_encoder.forward_stackgru.initStack()

	if(smiles_encoder.numdir==1):
		for k in range(len(input_tensor)):
			hidden_forward = smiles_encoder.forward_unidir(input_tensor[:,k], hidden_forward, stack_forward)
		mean, logvar = smiles_encoder.post_gru_reshape_function_unidir(hidden_forward)
		smiles_latent_vector = reparameterization_trick(mean, logvar)  #[1, batch_size, z_dim]

	else:
		hidden_backward = smiles_encoder.backward_stackgru.initHidden()
		stack_backward = smiles_encoder.backward_stackgru.initStack()

		backward_index = len(input_tensor) - 1
		for k in range(len(input_tensor)):
			hidden_forward, hidden_backward = smiles_encoder.forward_bidir(input_tensor[:,k], input_tensor[:,backward_index], hidden_forward, stack_forward, hidden_backward, stack_backward)
			backward_index = backward_index - 1
		mean, logvar = smiles_encoder.post_gru_reshape_function_bidir(hidden_forward, hidden_backward)
		smiles_latent_vector = reparameterization_trick(mean, logvar)  #[1, batch_size, z_dim]

	latent_vector = smiles_latent_vector + graph_latent_z  #Only possible if both tensors have same latent dimensions

	return latent_vector


#--------------------------------------------------------SAMPLING FUNCTION----------------------------------------------------------
def sampling(svae_encoder, svae_decoder, gvae_encoder, X_adj, X_features, batch_size, zdim, max_length, prime_str='!', end_token='E'):
	svae_encoder.eval()
	svae_decoder.eval()
	gvae_encoder.eval()

	prefix = torch.tensor(char_to_int["!"], device=device)
	prefix = prefix.unsqueeze(0).to(torch.int64)
	prefix = prefix.unsqueeze(0).to(torch.int64)

	#smiles_latent_z = torch.randn(1, batch_size, zdim).to(device)
	latent_vector = latent_vector = extract_latent_vector_v1(svae_encoder, gvae_encoder, X_adj, X_features, prefix)

	#Preparing the hidden state for the SMILES VAE decoder
	decoder_hidden = svae_decoder.latent_vector_to_hidden(latent_vector)
	decoder_stack = svae_decoder.decoder_stackgru.initStack()

	smiles = ""

        #Use priming string to "build up" hidden state
	_, decoder_hidden, decoder_stack = svae_decoder(prefix, decoder_hidden, decoder_stack) #Forward step gives output, hidden, stack - Acts like the encoder

	for p in range(max_length):
		output, decoder_hidden, decoder_stack = svae_decoder(prefix, decoder_hidden, decoder_stack)

		# Sample from the network as a multinomial distribution
		probs = torch.softmax(output, dim=2) #Apply softmax on output of LSTM/GRU
		top_i = torch.multinomial(probs.view(-1), 1)[0].cpu() #Using random sampling from the multinomial distribution to get the next character
		#print(torch.multinomial(probs.view(-1), 1).item())

		# Add predicted character to string and use as next input
		top_i = top_i.item()
		predicted_char = int_to_char[top_i]

		if predicted_char == end_token:
			break
		else:
			smiles += predicted_char
			prefix = torch.tensor(char_to_int[predicted_char], device=device)
			prefix = prefix.unsqueeze(0).to(torch.int64)

	return smiles


#-------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------FUNCTION CALL STATEMENTS-----------------------------------------------------
#Tokenize the SMILES to pass to the DataLoader
charset, char_to_int, int_to_char = prepare_dicts_and_dictfiles(char_to_int_file, int_to_char_file)
print("SMILES_vocabulary:", str(charset))

batch_size = 1
embed = 100   #Length to which all the SMILES have to be padded before passing into the model

#SMILES VAE parameters
latent_dim = 256
gru_dim = 1024
vocab_size = len(charset)
stack_depth = embed  #Need not have depth more than the maximum SMILES length in the dataset
stack_width = 256
kl_growth_rate = 0.05

#GCN VAE parameters
feature_dim = 9  #No. of features per node in all the networks
latent_dim = 256
gcn_hidden_dim = 128

#Creating the model objects for loading weights
smiles_encoder = VAE_Encoder(vocab_size, gru_dim, latent_dim, vocab_size, batch_size, stack_depth, stack_width, dropout=0.2, n_layers=2, bidir=True).to(device)
smiles_decoder = VAE_Decoder(vocab_size, gru_dim, latent_dim, vocab_size, batch_size, stack_depth, stack_width, dropout=0.2, n_layers=2, bidir=False).to(device)
gcn_encoder = GAT_VAE_Encoder(feature_dim, gcn_hidden_dim, latent_dim, dropout=0.2).to(device)

#Loading the best model weights on the model objects
state_old = torch.load(combined_cptfile, map_location=device)
smiles_encoder.load_state_dict(state_old['encoder_state_dict'])
smiles_decoder.load_state_dict(state_old['decoder_state_dict'])
gcn_encoder.load_state_dict(state_old['graph_encoder_state_dict'])

print("Model weights loaded...")

#Graph pre-processing - TODO: Consider multiple 
A, prot_X = load_data(protein_index, graphpath)
prot_A = preprocess_graph(A)
prot_A = prot_A.unsqueeze(0)
prot_X = prot_X.unsqueeze(0)


#-----------------------------------------------------SAMPLING BEGINS HERE--------------------------------------------------------------
nmol = 1000
correctmol=[]
rdkitmols=[]
wrong = 0
correct = 0

for i in range(nmol):
	length = 0
	while length==0:
		try:
			output_seq = sampling(smiles_encoder, smiles_decoder, gcn_encoder, prot_A, prot_X, batch_size, latent_dim, embed, prime_str='!', end_token='E')
			rdkitmol=Chem.MolFromSmiles(output_seq) #Checks if the SMILES can be converted into a chemically valid molecule
			if rdkitmol:
				length = 1
				output_seq = Chem.MolToSmiles(rdkitmol)  #Canonicalization
				#print("Correct:",output_seq)
				correctmol.append(output_seq)
				rdkitmols.append(rdkitmol)
				correct = correct + 1
			else:
				#print("Wrong:",output_seq)
				wrong=wrong+1
				length = 0
		except:
				continue

print(protein_index,"protein: ",(wrong/float(correct+wrong)*100)," % wrongly formatted smiles!")

#outfile = savepath+prot+"_sampling_test.smi"
#Write the correct molecules to an outfile
with open(outfile, 'w') as f:
	for molecule in correctmol:
			print(molecule, file=f)











