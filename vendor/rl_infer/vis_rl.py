#SMILES Generative model - RNN encoder-decoder architecture - PyTorch implementation

import sys
import csv
import os
import re
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
import shutil
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from math import sqrt
from rdkit import Chem
from scipy.spatial.distance import cdist
from itertools import product
from rdkit.Chem import AllChem, QED
from rdkit.Chem import rdMolDescriptors    ##new addition
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from sascorer import readFragmentScores, numBridgeheadsAndSpiro, calculateScore   ##new addition
from io import open
from sascorer import readFragmentScores, numBridgeheadsAndSpiro, calculateScore
from pic50_scorer import readModel as readPic50Model, calculateScore as calculatePic50
from sol_scorer import readModel as readSolModel, calculateScore as calculateSolubility
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
if len(sys.argv) != 10:
	print(f"ERROR: vis_rl.py expects 9 arguments, got {len(sys.argv) - 1}")
	print("Usage: python vis_rl.py indexfile char_to_int int_to_char graphpath "
	      "svae_gvae_cpt predictor_cpt docking_path savepath retraining_flag")
	sys.exit(1)

indexfile = sys.argv[1]
char_to_int_file = sys.argv[2]
int_to_char_file = sys.argv[3]
graphpath = sys.argv[4]
svae_gvae_combined_cptfile = sys.argv[5]  ##Pre-trained weight
predictor_cptfile = sys.argv[6]   ##DTA- weight files
docking_path = sys.argv[7]  ##gnina or vina
savepath = sys.argv[8]
retraining_flag = sys.argv[9]
if not docking_path.endswith("/"):
	docking_path = docking_path + "/"
if not savepath.endswith("/"):
	savepath = savepath + "/"

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
	return one_hot
	#return one_hot[:,0:-1], one_hot[:,1:] #Input has -1 to indicate all columns from 0 except the last column, Output has all columns except the ! mark in the first column
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
		return torch.zeros(self.batch_size, self.stack_depth, self.stack_width, device=device)


#----------------------------------------------------VAE ENCODER DEFINITION--------------------------------------------------------
class VAE_Encoder(nn.Module):
	def __init__(self, input_size, hidden_size, z_dim, output_size, batch_size, stack_depth, stack_width, dropout=0.2, n_layers=1, bidir=False):
		super(VAE_Encoder, self).__init__()

		self.z_dim = z_dim
		self.hidden_size = hidden_size
		self.num_layers = n_layers
		self.stack_depth = stack_depth
		self.stack_width = stack_width
		self.bidir = bidir
		self.batch_size = batch_size
		self.numdir=1

		self.forward_stackgru = Stack_GRU(input_size, hidden_size, output_size, stack_depth, stack_width, self.batch_size, dropout=dropout, n_layers=n_layers, bidir=False)

		if(self.bidir):
			self.numdir=2
			self.backward_stackgru = Stack_GRU(input_size, hidden_size, output_size, stack_depth, stack_width, self.batch_size, dropout=dropout, n_layers=n_layers, bidir=False)

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
		self.decoder_stackgru = Stack_GRU(input_size, hidden_size, output_size, stack_depth, stack_width, self.batch_size, dropout=dropout, n_layers=n_layers, bidir=False)  #By default the decoder is always unidirectional - Bidirectional decoder has not yet been implemented

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


#--------------------------------------------------LATENT VECTOR EXTRACTION FUNCTION-------------------------------------------------
#For the GRAPH + SMILES model
def extract_latent_vector_v1(smiles_encoder, gat_encoder, X_adj, X_feat, prefix):
	#Preparing the graph latent vector
	X_adj = X_adj.to(device)
	X_feat = X_feat.to(device)
	graph_mean, graph_logvar = gat_encoder([X_feat], [X_adj])  #Both tensors are [No. of nodes, 256] dimension
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

#For the DTA predictive model
def extract_latent_vector_v2(encoder, gvae_encoder, X_adj, X_feat, smiles):
	smiles_tokenized = []
	smiles_tokenized.append(tokenize(smiles))
	smiles_tokenized = np.array(smiles_tokenized)
	input_tensor = prepare_dataset(smiles_tokenized)  #One-hot encoding the SMILES
	input_tensor = torch.from_numpy(input_tensor).to(device) #[1, embed]
	#input_tensor = input_tensor.unsqueeze(0)

	hidden_gru = encoder.initHidden()

	smiles_latent_vector = encoder(input_tensor, hidden_gru)
	smiles_latent_vector = torch.mean(smiles_latent_vector, dim=1)
	smiles_latent_vector = smiles_latent_vector.unsqueeze(0)  #Becomes [1,1,256]

	#Combine the SMILES latent vector and input graph latent vector
	X_adj = X_adj.to(device)
	X_feat = X_feat.to(device)
	graph_latent_vector = gvae_encoder(X_feat, X_adj)
	#graph_mean, graph_logvar = gvae_encoder(X_feat, X_adj)  #Both tensors are [No. of nodes, 256] dimension
	#graph_latent_vector = reparameterization_trick_graph(graph_mean, graph_logvar)  #[No. of nodes, 256]
	graph_latent_vector = torch.mean(graph_latent_vector, dim=0)  #[256] - Will average over the node embeddings to get a pooled vector
	graph_latent_vector = graph_latent_vector.unsqueeze(0)  #[1, 256] - To add a batch_size dimension
	graph_latent_vector = graph_latent_vector.unsqueeze(0)  #[1, 1, 256]

	#print(smiles_latent_vector.size(), graph_latent_vector.size()) #[1,256] [1,1,128]

	#latent_vector = smiles_latent_vector + graph_latent_vector
	latent_vector = torch.cat((smiles_latent_vector, graph_latent_vector), dim=-1)  #[1,1,384] = [1,1,256+128]
	#print(latent_vector.size())

	return latent_vector


#--------------------------------------------------------SAMPLING FUNCTION----------------------------------------------------------
def sampling(svae_encoder, svae_decoder, gvae_encoder, X_adj, X_features, max_length, prime_str='!', end_token='E'):
	#svae_encoder.eval()
	#svae_decoder.eval()
	#gvae_encoder.eval()

	prefix = torch.tensor(char_to_int["!"], device=device)
	prefix = prefix.unsqueeze(0).to(torch.int64)
	prefix = prefix.unsqueeze(0).to(torch.int64)

	#Extract the combined latent vector = graph + "!"
	latent_vector = extract_latent_vector_v1(svae_encoder, gvae_encoder, X_adj, X_features, prefix)

	#Preparing the hidden state for the SMILES VAE decoder
	decoder_hidden = svae_decoder.latent_vector_to_hidden(latent_vector)
	decoder_stack = svae_decoder.decoder_stackgru.initStack()

	smiles = ""
	probs_out = []
	top_index = []

		#Use priming string to "build up" hidden state
	_, decoder_hidden, decoder_stack = svae_decoder(prefix, decoder_hidden, decoder_stack)  #TODO: Check if this is required!

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
			probs_out.append(output)
			top_index.append(top_i)
			break
		else:
			smiles += predicted_char
			probs_out.append(output)
			top_index.append(top_i)
			prefix = []
			prefix = torch.tensor(char_to_int[predicted_char], device=device)
			prefix = prefix.unsqueeze(0).to(torch.int64)

	svae_encoder.train()
	svae_decoder.train()
	gvae_encoder.train()
	return smiles, probs_out, top_index


#--------------------------------------------------PRIOR LIKELIHOOD CALCULATION FUNCTION-------------------------------------------------
#Given a model/policy calculate the likelihood of a SMILES to be generated using that model/policy
def prior_likelihood(smiles_encoder, smiles_decoder, graph_encoder, X_adj, X_feat, smiles):
	smiles_encoder.eval()
	smiles_decoder.eval()

	#Get the latent vector of the smiles from the encoder
	prefix = torch.tensor(char_to_int["!"], device=device)
	prefix = prefix.unsqueeze(0).to(torch.int64)
	prefix = prefix.unsqueeze(0).to(torch.int64)
	latent_vector = extract_latent_vector_v1(smiles_encoder, graph_encoder, X_adj, X_feat, prefix)

	#Converting string of characters into tensor
	smiles = "!"+smiles+"E"
	smiles_tokenized = tokenize(smiles)
	smiles_chars = smiles_tokenized.split("*")

	decoder_stack = smiles_decoder.decoder_stackgru.initStack()
	decoder_hidden = smiles_decoder.latent_vector_to_hidden(latent_vector)  #Get the initial decoder hidden state from the latent vector

	top_in = []
	prior_likelihood = 0.0

	#"Following" the trajectory and accumulating the loss for subsequent characters
	for i in range(len(smiles_chars)-1):   #First character probability is computed before the loop 
		prefix = []
		prefix.append(char_to_int[smiles_chars[i]])
		prefix = torch.tensor(prefix, device=device)
		prefix = prefix.unsqueeze(0).to(torch.int64)

		output, decoder_hidden, decoder_stack = smiles_decoder(prefix, decoder_hidden, decoder_stack)
		log_probs = F.log_softmax(output, dim=-1)
		top_i = char_to_int[smiles_chars[i+1]]
		#print("Prior:",log_probs,"\nChar:",top_i)
		top_in.append(top_i)
		#print("Character:",smiles_chars[i])
		action_prob = log_probs[0,0,top_i].item()
		#print("Prior action prob:",action_prob)
		prior_likelihood = prior_likelihood + action_prob  #Product of the action probabilities is the model likelihood

	#print("Prior:",top_in)

	smiles_encoder.train()
	smiles_decoder.train()
	return prior_likelihood


#-------------------------------------------------------USING GNINA FOR DOCKING----------------------------------------------------
def generate_complex_structure_and_descriptors(smiles, molcount, output_path, filepath):
	config = filepath+"config.txt"
	gnina = filepath+"gnina"
	fname = "mol"+str(molcount)
	print(smiles)
	rdkitmol = Chem.MolFromSmiles(smiles)
	mol2 = Chem.AddHs(rdkitmol)
	AllChem.EmbedMolecule(mol2, randomSeed=0xf00d)
	AllChem.MMFFOptimizeMolecule(mol2)
	# print(Chem.MolToMolBlock(mol2), file=open(output_path+fname+".mol",'w+'))
	with Chem.SDWriter(output_path+fname+".sdf") as out:
		out.write(mol2)

	# os.system("obabel -imol "+output_path+fname+".mol -osdf -O "+output_path+fname+".sdf")
	os.system(gnina+" --config "+config+" --ligand "+output_path+fname+".sdf --out "+output_path+fname+"_out.sdf --log "+output_path+fname+"_log.txt --cnn_scoring none")
	#os.system("obabel -ipdbqt "+output_path+fname+"_out.pdbqt -osdf -O "+output_path+fname+"_out.sdf")

	#---------------------------------------------------------------------------------------------------------------
	# Possible predefined protein atoms
	ECIF_ProteinAtoms = ['C;4;1;3;0;0', 'C;4;2;1;1;1', 'C;4;2;2;0;0', 'C;4;2;2;0;1', 'C;4;3;0;0;0', 'C;4;3;0;1;1', 'C;4;3;1;0;0', 'C;4;3;1;0;1', 'C;5;3;0;0;0', 'C;6;3;0;0;0', 'N;3;1;2;0;0', 'N;3;2;0;1;1', 'N;3;2;1;0;0', 'N;3;2;1;1;1', 'N;3;3;0;0;1', 'N;4;1;2;0;0', 'N;4;1;3;0;0', 'N;4;2;1;0;0', 'O;2;1;0;0;0', 'O;2;1;1;0;0', 'S;2;1;1;0;0', 'S;2;2;0;0;0']

	# Possible ligand atoms according to the PDBbind 2016 "refined set"
	ECIF_LigandAtoms = ['Br;1;1;0;0;0', 'C;3;3;0;1;1', 'C;4;1;1;0;0', 'C;4;1;2;0;0', 'C;4;1;3;0;0', 'C;4;2;0;0;0', 'C;4;2;1;0;0', 'C;4;2;1;0;1', 'C;4;2;1;1;1', 'C;4;2;2;0;0', 'C;4;2;2;0;1', 'C;4;3;0;0;0', 'C;4;3;0;0;1', 'C;4;3;0;1;1', 'C;4;3;1;0;0', 'C;4;3;1;0;1', 'C;4;4;0;0;0', 'C;4;4;0;0;1', 'C;5;3;0;0;0', 'C;5;3;0;1;1', 'C;6;3;0;0;0', 'Cl;1;1;0;0;0', 'F;1;1;0;0;0', 'I;1;1;0;0;0', 'N;3;1;0;0;0', 'N;3;1;1;0;0', 'N;3;1;2;0;0', 'N;3;2;0;0;0', 'N;3;2;0;0;1', 'N;3;2;0;1;1', 'N;3;2;1;0;0', 'N;3;2;1;0;1', 'N;3;2;1;1;1', 'N;3;3;0;0;0', 'N;3;3;0;0;1', 'N;3;3;0;1;1', 'N;4;1;2;0;0', 'N;4;1;3;0;0', 'N;4;2;1;0;0', 'N;4;2;2;0;0', 'N;4;2;2;0;1', 'N;4;3;0;0;0', 'N;4;3;0;0;1', 'N;4;3;1;0;0', 'N;4;3;1;0;1', 'N;4;4;0;0;0', 'N;4;4;0;0;1', 'N;5;2;0;0;0', 'N;5;3;0;0;0', 'N;5;3;0;1;1', 'O;2;1;0;0;0', 'O;2;1;1;0;0', 'O;2;2;0;0;0', 'O;2;2;0;0;1', 'O;2;2;0;1;1', 'P;5;4;0;0;0', 'P;6;4;0;0;0', 'P;6;4;0;0;1', 'P;7;4;0;0;0', 'S;2;1;0;0;0', 'S;2;1;1;0;0', 'S;2;2;0;0;0', 'S;2;2;0;0;1', 'S;2;2;0;1;1', 'S;3;3;0;0;0', 'S;3;3;0;0;1', 'S;4;3;0;0;0', 'S;6;4;0;0;0', 'S;6;4;0;0;1', 'S;7;4;0;0;0']

	PossibleECIF = [i[0]+"-"+i[1] for i in product(ECIF_ProteinAtoms, ECIF_LigandAtoms)]
	ELEMENTS_ProteinAtoms = ["C","N","O", "S"]
	ELEMENTS_LigandAtoms = ["Br", "C", "Cl", "F", "I", "N", "O", "P", "S"]
	PossibleELEMENTS = [i[0]+"-"+i[1] for i in product(ELEMENTS_ProteinAtoms, ELEMENTS_LigandAtoms)]

	LigandDescriptors = ['MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex', 'MinAbsEStateIndex', 'qed', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA14', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount', 'MolLogP', 'MolMR', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_bicyclic', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_urea']

	DescCalc = MolecularDescriptorCalculator(LigandDescriptors)
	
	def GetAtomType(atom):
	# This function takes an atom in a molecule and returns its type as defined for ECIF
		
		AtomType = [atom.GetSymbol(), str(atom.GetExplicitValence()), str(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() != "H"])), str(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() == "H"])), str(int(atom.GetIsAromatic())), str(int(atom.IsInRing())), ]

		return(";".join(AtomType))

	def LoadSDFasDF(SDF):
	# This function takes an SDF for a ligand as input and returns it as a pandas DataFrame with its atom types labeled according to ECIF
		
		m = Chem.MolFromMolFile(SDF, sanitize=False)
		m.UpdatePropertyCache(strict=False)
		
		ECIF_atoms = []

		for atom in m.GetAtoms():
			if atom.GetSymbol() != "H": # Include only non-hydrogen atoms
				entry = [int(atom.GetIdx())]
				entry.append(GetAtomType(atom))
				pos = m.GetConformer().GetAtomPosition(atom.GetIdx())
				entry.append(float("{0:.4f}".format(pos.x)))
				entry.append(float("{0:.4f}".format(pos.y)))
				entry.append(float("{0:.4f}".format(pos.z)))
				ECIF_atoms.append(entry)

		df = pd.DataFrame(ECIF_atoms)
		df.columns = ["ATOM_INDEX", "ECIF_ATOM_TYPE","X","Y","Z"]
		if len(set(df["ECIF_ATOM_TYPE"]) - set(ECIF_LigandAtoms)) > 0:
			print("WARNING: Ligand contains unsupported atom types. Only supported atom-type pairs are counted.")

		return(df)


	Atom_Keys=pd.read_csv(filepath+"PDB_Atom_Keys.csv", sep=",")
	def LoadPDBasDF(PDB):
	# This function takes a PDB for a protein as input and returns it as a pandas DataFrame with its atom types labeled according to ECIF
		ECIF_atoms = []
		f = open(PDB)
		for i in f:
			if i[:4] == "ATOM":
			# Include only non-hydrogen atoms
				if (len(i[12:16].replace(" ","")) < 4 and i[12:16].replace(" ","")[0] != "H") or (len(i[12:16].replace(" ","")) == 4 and i[12:16].replace(" ","")[1] != "H" and i[12:16].replace(" ","")[0] != "H"):
					ECIF_atoms.append([int(i[6:11]), i[17:20]+"-"+i[12:16].replace(" ",""), float(i[30:38]), float(i[38:46]), float(i[46:54])])
		f.close()
		
		df = pd.DataFrame(ECIF_atoms, columns=["ATOM_INDEX","PDB_ATOM","X","Y","Z"])
		df = df.merge(Atom_Keys, left_on='PDB_ATOM', right_on='PDB_ATOM')[["ATOM_INDEX", "ECIF_ATOM_TYPE", "X", "Y", "Z"]].sort_values(by="ATOM_INDEX").reset_index(drop=True)
		if list(df["ECIF_ATOM_TYPE"].isna()).count(True) > 0:
			print("WARNING: Protein contains unsupported atom types. Only supported atom-type pairs are counted.")

		return(df)


	def GetPLPairs(PDB_protein, SDF_ligand, distance_cutoff=6.0):
	# This function returns the protein-ligand atom-type pairs for a given distance cutoff
		
		# Load both structures as pandas DataFrames
		Target = LoadPDBasDF(PDB_protein)
		Ligand = LoadSDFasDF(SDF_ligand)
		
		# Take all atoms from the target within a cubic box around the ligand considering the "distance_cutoff criterion"
		for i in ["X","Y","Z"]:
			Target = Target[Target[i] < float(Ligand[i].max())+distance_cutoff]
			Target = Target[Target[i] > float(Ligand[i].min())-distance_cutoff]
		
		# Get all possible pairs
		Pairs = list(product(Target["ECIF_ATOM_TYPE"], Ligand["ECIF_ATOM_TYPE"]))
		Pairs = [x[0]+"-"+x[1] for x in Pairs]
		Pairs = pd.DataFrame(Pairs, columns=["ECIF_PAIR"])
		Distances = cdist(Target[["X","Y","Z"]], Ligand[["X","Y","Z"]], metric="euclidean")
		Distances = Distances.reshape(Distances.shape[0]*Distances.shape[1],1)
		Distances = pd.DataFrame(Distances, columns=["DISTANCE"])

		Pairs = pd.concat([Pairs,Distances], axis=1)
		Pairs = Pairs[Pairs["DISTANCE"] <= distance_cutoff].reset_index(drop=True)
		# Pairs from ELEMENTS could be easily obtained froms pairs from ECIF
		Pairs["ELEMENTS_PAIR"] = [x.split("-")[0].split(";")[0]+"-"+x.split("-")[1].split(";")[0] for x in Pairs["ECIF_PAIR"]]
		return Pairs


	def GetECIF(PDB_protein, SDF_ligand, distance_cutoff=6.0):
	# Main function for the calculation of ECIF
		Pairs = GetPLPairs(PDB_protein, SDF_ligand, distance_cutoff=distance_cutoff)
		ECIF = [list(Pairs["ECIF_PAIR"]).count(x) for x in PossibleECIF]
		return ECIF


	def GetELEMENTS(PDB_protein, SDF_ligand, distance_cutoff=6.0):
	# Function for the calculation of ELEMENTS
		Pairs = GetPLPairs(PDB_protein, SDF_ligand, distance_cutoff=distance_cutoff)
		ELEMENTS = [list(Pairs["ELEMENTS_PAIR"]).count(x) for x in PossibleELEMENTS]
		return ELEMENTS


	def GetRDKitDescriptors(SDF):
	# Function for the calculation of ligand descriptors
		mol = Chem.MolFromMolFile(SDF, sanitize=False)
		mol.UpdatePropertyCache(strict=False)
		Chem.GetSymmSSSR(mol)
		return DescCalc.CalcDescriptors(mol)
	#-------------------------------------------------------------------------------------------------------------

	receptor = filepath+"receptor.pdb"
	ligand = output_path+fname+"_out.sdf"
	ECIF = GetECIF(receptor, ligand, distance_cutoff=6.0)
	descriptors = list(GetRDKitDescriptors(ligand))

	ECIF_desc_combined = []
	ECIF_desc_combined.extend(ECIF)
	ECIF_desc_combined.extend(descriptors)
	ECIF_desc_combined = np.asarray(ECIF_desc_combined)
	ECIF_desc_combined = ECIF_desc_combined.reshape(1, -1)
	
	return ECIF_desc_combined


#----------------------------------------------------DOCKING AND SCORE EXTRACTION-------------------------------------------------------
def perform_docking(smiles, molcount, output_path, filepath):
	config = docking_path+"config.txt"
	smina = docking_path+"smina.static"
	fname = "mol"+str(molcount)
	print(smiles)
	rdkitmol = Chem.MolFromSmiles(smiles)
	mol2 = Chem.AddHs(rdkitmol)
	AllChem.EmbedMolecule(mol2, randomSeed=0xf00d)
	AllChem.MMFFOptimizeMolecule(mol2)
	# print(Chem.MolToMolBlock(mol2), file=open(output_path+fname+".mol",'w+'))
	with Chem.SDWriter(output_path+fname+".sdf") as out:
		out.write(mol2)

	# os.system("obabel -imol "+output_path+fname+".mol -osdf -O "+output_path+fname+".sdf")
	os.system(smina+" --config "+config+" --ligand "+output_path+fname+".sdf --out "+output_path+fname+"_out.sdf --log "+output_path+fname+"_log.txt")
	
	return output_path+fname+"_log.txt"
	
def extract_docking_score(docking_outfile):
	with open(docking_outfile) as dfile:
		for line in dfile.readlines():
			line = line.strip()
			if(re.search("^\d{1}\s+", line)):
				matches = re.search("\d{1}\s+(.{5})\s+.{1,5}\s+.{1,5}", line)
				dscore = float(matches.group(1))
				break
	return dscore
#------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------RL REWARD FUNCTION----------------------------------------------------------

def geometric_mean_rewards(rewards):
    if not rewards:
        return 0.0
    vals = np.array(rewards, dtype=np.float64)
    if not np.all(np.isfinite(vals)) or np.any(vals <= 0):
        return 0.0
    return float(np.exp(np.mean(np.log(vals))))


def init_scorers():
    """Call once before RL training (same as readFragmentScores() in original script)."""
    readFragmentScores()
    readPic50Model()
    readSolModel()


#def get_reward(ecif_input, smiles, predictor): #Specific to logP - Based on Popova et al
def get_reward(docking_outfile, smiles, predictor):
    reward1 = 0
    dockscore = extract_docking_score(docking_outfile)
    reward1 = math.exp(-dockscore / 3.0)

    rdkitmol = Chem.MolFromSmiles(smiles)
    if rdkitmol is None:
        return 0.0

    ##New Modification — SA (original)
    reward2 = 0
    sas = calculateScore(rdkitmol)
    reward2 = math.exp(sas / 3.0)

    reward3 = 0
    clogp = Descriptors.MolLogP(rdkitmol)
    if clogp >= -1 and clogp <= 3:
        reward3 = 11
    else:
        reward3 = 1

    ##New Modification — MW band (same style as logP)
    reward4 = 0
    mw = Descriptors.MolWt(rdkitmol)
    if mw >= 380 and mw <= 810:
        reward4 = 11
    else:
        reward4 = 1

    ##New Modification — QED (same exp style as SA)
    reward5 = 0
    qed = QED.qed(rdkitmol)
    reward5 = math.exp(qed / 0.3)

    ##New Modification — pIC50 (same exp style as SA; raw from pic50_scorer)
    reward6 = 0
    pic50 = calculatePic50(smiles)
    if math.isfinite(pic50):
        reward6 = math.exp(pic50 / 3.0)
    else:
        reward6 = 0

    ##New Modification — solubility logS (shifted exp, same pattern)
    reward7 = 0
    sol = calculateSolubility(smiles)
    if math.isfinite(sol):
        reward7 = math.exp((sol - (-13.17)) / 3.0)
    else:
        reward7 = 0

    reward = 0
    reward = geometric_mean_rewards([reward1, reward2, reward3, reward4, reward5, reward6, reward7])
    return reward


#-------------------------------------------------------RL POLICY GRADIENT UPDATE FUNCTION------------------------------------------
def policy_gradient(X_adj, X_features, prior_svae_encoder, prior_svae_decoder, prior_gvae_encoder, agent_svae_encoder, agent_svae_decoder, agent_gvae_encoder, optimizer, predictor, output_path, filepath, char_to_int, int_to_char, max_len, nmols=256, sigma=3):
    rl_loss = 0
    optimizer.zero_grad()
    total_reward = 0
    avg_reward = 0
    batch_gen_itercount = 0

    trajectory_tensor = []
    graph_adj_tensor = []
    graph_feat_tensor = []
    traj_probs_tensor = []
    top_indices_tensor = []
    reward_tensor = []
    baseline_reward = 0
    molcount = 0

    #Make sure the DataLoader shuffle=True so that the batches are randomized for each run
    #for inputbatch, targetbatch, X_adj, X_features, Y_adj, Y_features, pos_weight, norm in train_data:
    for i in range(nmols):
        reward = 0
        while reward == 0:
            batch_gen_itercount = batch_gen_itercount + 1
            trajectory, traj_probs, top_indices = sampling(agent_svae_encoder, agent_svae_decoder, agent_gvae_encoder, X_adj, X_features, max_len, prime_str='!', end_token='E')
            #Try-except block to handle bad conformer exception from RDKit Mol2MolBlock function
            try:
                mol = Chem.MolFromSmiles(trajectory)  #Check if molecule is valid
                if mol:
                    trajectory_tensor.append(trajectory)
                    traj_probs_tensor.append(traj_probs)
                    top_indices_tensor.append(top_indices)

                    docking_outfile = perform_docking(trajectory, batch_gen_itercount, output_path, filepath)
                    reward = get_reward(docking_outfile, trajectory, predictor)
                    print(trajectory, reward)
                    reward_tensor.append(reward)
                    graph_adj_tensor.append(X_adj)
                    graph_feat_tensor.append(X_features)
                else:
                    reward = 0  #Invalid molecules get a reward of 0
            except:
                reward = 0
                continue

    for i, reward in enumerate(reward_tensor):
        baseline_reward = baseline_reward + reward
    baseline_reward = baseline_reward / nmols

    for i in range(nmols):
        trajectory = trajectory_tensor[i]
        traj_probs = traj_probs_tensor[i]
        top_indices = top_indices_tensor[i]
        reward = reward_tensor[i]

        trajectory_tokenized = tokenize(trajectory)
        trajectory_chars = trajectory_tokenized.split("*")

        total_reward += reward
        prior_llh = prior_likelihood(prior_svae_encoder, prior_svae_decoder, prior_gvae_encoder, graph_adj_tensor[i], graph_feat_tensor[i], trajectory)
        augmented_llh = prior_llh + reward * sigma

        agent_llh = 0.0
        for i in range(len(traj_probs)):
            output = traj_probs[i]
            log_probs = F.log_softmax(output, dim=-1)
            action_prob = log_probs[0, 0, top_indices[i]]
            agent_llh = agent_llh + action_prob

        returns = 0.0
        returns = -((augmented_llh - agent_llh) ** 2)
        rl_loss = rl_loss + (-returns)

    rl_loss = rl_loss / nmols
    avg_reward = total_reward / nmols

    agent_svae_encoder.train()
    agent_svae_decoder.train()
    agent_gvae_encoder.train()
    prior_svae_encoder.train()
    prior_gvae_encoder.train()
    prior_svae_decoder.train()

    rl_loss.backward()
    nn.utils.clip_grad_norm_(list(agent_svae_decoder.parameters()) + list(agent_gvae_encoder.parameters()), 3)
    optimizer.step()

    return avg_reward, rl_loss.item(), batch_gen_itercount


#-------------------------------------------------RL POLICY GRADIENT TRAINING FUNCTION------------------------------------------------
def rltrainingloop(prot_A, prot_X, prior_svae_encoder, prior_svae_decoder, prior_gvae_encoder, agent_svae_encoder, agent_svae_decoder, agent_gvae_encoder, optimizer, predictor, output_path, char_to_int, int_to_char, max_length, batch_size, n_iters=100, savecpt=1, sigma_val=3, cp=0):
    total_rl_loss = []
    avg_rl_reward = []

    if cp == 0:
        for iteration in range(0, n_iters):
            filepath = output_path
            dirpath = os.path.join(output_path, "RL_iter_" + str(iteration))
            os.mkdir(dirpath)
            dirpath = dirpath + "/"

            avg_reward, rl_loss, batchitercount = policy_gradient(prot_A, prot_X, prior_svae_encoder, prior_svae_decoder, prior_gvae_encoder, agent_svae_encoder, agent_svae_decoder, agent_gvae_encoder, optimizer, predictor, dirpath, filepath, char_to_int, int_to_char, max_length, nmols=batch_size, sigma=sigma_val)
            print("Iter:", (iteration + 1), "Average reward:", avg_reward, "RL loss:", rl_loss, "Sampling iterations:", batchitercount)
            total_rl_loss.append(rl_loss)
            avg_rl_reward.append(avg_reward)
            if ((iteration + 1) % savecpt) == 0:
                path = savepath + "RL_Weights_epoch_" + str(iteration + 1) + ".cpt"
                state = {'epoch': iteration + 1, 'encoder_state_dict': agent_svae_encoder.state_dict(), 'decoder_state_dict': agent_svae_decoder.state_dict(), 'graph_encoder_state_dict': agent_gvae_encoder.state_dict(), 'optimizer': optimizer.state_dict(), 'rl_loss': rl_loss, 'avg_reward': avg_reward, 'batchitercount': batchitercount}
                torch.save(state, path)
                output_path = filepath
            else:
                shutil.rmtree(filepath + "RL_iter_" + str(iteration))
                output_path = filepath

    else:
        combined_state_old = torch.load(svae_gvae_combined_cptfile, map_location=device)
        prior_smiles_encoder.load_state_dict(combined_state_old['encoder_state_dict'])
        prior_smiles_decoder.load_state_dict(combined_state_old['decoder_state_dict'])
        agent_smiles_encoder.load_state_dict(combined_state_old['encoder_state_dict'])
        agent_smiles_decoder.load_state_dict(combined_state_old['decoder_state_dict'])

        prior_gcn_encoder.load_state_dict(combined_state_old['graph_encoder_state_dict'])
        agent_gcn_encoder.load_state_dict(combined_state_old['graph_encoder_state_dict'])
        optimizer.load_state_dict(combined_state_old['optimizer'])

        print("Old model weights loaded...")
        print("Training continued...")

        iteration_old = combined_state_old['epoch']
        for iteration in range(iteration_old + 1, n_iters):
            filepath = output_path
            dirpath = os.path.join(output_path, "RL_iter_" + str(iteration))
            os.mkdir(dirpath)
            dirpath = dirpath + "/"

            avg_reward, rl_loss, batchitercount = policy_gradient(prot_A, prot_X, prior_svae_encoder, prior_svae_decoder, prior_gvae_encoder, agent_svae_encoder, agent_svae_decoder, agent_gvae_encoder, optimizer, predictor, dirpath, filepath, char_to_int, int_to_char, max_length, nmols=batch_size, sigma=sigma_val)
            print("Iter:", (iteration + 1), "Average reward:", avg_reward, "RL loss:", rl_loss, "Sampling iterations:", batchitercount)
            total_rl_loss.append(rl_loss)
            avg_rl_reward.append(avg_reward)
            if ((iteration + 1) % savecpt) == 0:
                path = savepath + "RL_Weights_epoch_" + str(iteration + 1) + ".cpt"
                state = {'epoch': iteration + 1, 'encoder_state_dict': agent_svae_encoder.state_dict(), 'decoder_state_dict': agent_svae_decoder.state_dict(), 'graph_encoder_state_dict': agent_gvae_encoder.state_dict(), 'optimizer': optimizer.state_dict(), 'rl_loss': rl_loss, 'avg_reward': avg_reward, 'batchitercount': batchitercount}
                torch.save(state, path)
                output_path = filepath
            else:
                shutil.rmtree(filepath + "RL_iter_" + str(iteration))
                output_path = filepath

    return total_rl_loss, avg_rl_reward


#---------------------------------------------------------------------------------------------------------------------------------------
# MAIN
#---------------------------------------------------------------------------------------------------------------------------------------
def _resolve_graphpath(graph_path):
	"""Accept graph directory or a path ending in .graph / .x."""
	path = graph_path.rstrip("/")
	receptor_hint = None
	base = os.path.basename(path)
	if base.endswith(".graph"):
		receptor_hint = base[: -len(".graph")]
		path = os.path.dirname(path)
	elif base.endswith(".x"):
		receptor_hint = base[: -len(".x")]
		path = os.path.dirname(path)
	if path and not path.endswith("/"):
		path = path + "/"
	return path, receptor_hint


def _load_index(index_path):
	lower = index_path.lower()
	if lower.endswith(".pkl") or lower.endswith(".pickle"):
		with open(index_path, "rb") as handle:
			data = pickle.load(handle)
		if isinstance(data, dict):
			return data
		if isinstance(data, (list, tuple)) and data:
			return {"receptor": data[0]}
		raise ValueError("index pickle must be a dict or list")

	config = {}
	with open(index_path, "r", encoding="utf-8", errors="replace") as handle:
		lines = [ln.strip() for ln in handle.readlines() if ln.strip() and not ln.strip().startswith("#")]

	if not lines:
		raise ValueError(f"index file is empty: {index_path}")

	for line in lines:
		if "=" in line:
			key, value = line.split("=", 1)
			config[key.strip()] = value.strip()
			continue
		if "\t" in line:
			parts = [p.strip() for p in line.split("\t") if p.strip()]
			if parts and "receptor" not in config:
				config["receptor"] = parts[0]
			continue
		if "," in line:
			parts = [p.strip() for p in line.split(",") if p.strip()]
			if parts and "receptor" not in config:
				config["receptor"] = parts[0]
			continue
		if "receptor" not in config:
			config["receptor"] = line

	if "receptor" not in config:
		raise ValueError(f"could not parse receptor from index file: {index_path}")
	return config


def _infer_svae_encoder_hparams(encoder_sd):
	hidden_size = int(encoder_sd["forward_stackgru.embedding.weight"].shape[1])
	stack_width = int(encoder_sd["forward_stackgru.stack_input_layer.weight"].shape[0])
	z_dim = int(encoder_sd["mean.weight"].shape[0])
	bidir = "backward_stackgru.embedding.weight" in encoder_sd
	n_layers = sum(1 for k in encoder_sd if k.startswith("forward_stackgru.gru.weight_ih_l"))
	return hidden_size, stack_width, z_dim, bidir, n_layers


def _infer_gat_hparams(graph_sd):
	inp_feature_dim = int(graph_sd["attention_0.W"].shape[0])
	hidden_size = int(graph_sd["attention_0.W"].shape[1])
	z_dim = int(graph_sd["mean.W"].shape[1])
	return inp_feature_dim, hidden_size, z_dim


def _cfg_int(index_data, key, default):
	if key in index_data:
		return int(index_data[key])
	return default


def _cfg_float(index_data, key, default):
	if key in index_data:
		return float(index_data[key])
	return default


def run_training():
	global embed, char_to_int, int_to_char, graphpath

	print("Loading index:", indexfile)
	index_data = _load_index(indexfile)

	graphpath, graph_receptor = _resolve_graphpath(graphpath)
	if graph_receptor and not index_data.get("receptor"):
		index_data["receptor"] = graph_receptor

	receptor = index_data.get("receptor") or index_data.get("receptors", [None])[0]
	if receptor is None:
		raise ValueError("indexfile must contain 'receptor' or pass graphpath like .../9iow.graph")

	charset, char_to_int, int_to_char = prepare_dicts_and_dictfiles(char_to_int_file, int_to_char_file)
	input_size = len(char_to_int)
	output_size = len(char_to_int)

	print("Loading graph for receptor:", receptor, "from", graphpath)
	prot_A, prot_X = load_data(receptor, graphpath)
	prot_A = preprocess_graph(prot_A)
	inp_feature_dim = prot_X.shape[1]

	print("Loading checkpoint:", svae_gvae_combined_cptfile)
	combined_state = torch.load(svae_gvae_combined_cptfile, map_location=device)
	encoder_sd = combined_state["encoder_state_dict"]
	graph_sd = combined_state["graph_encoder_state_dict"]

	hidden_size, stack_width, z_dim, bidir, n_layers = _infer_svae_encoder_hparams(encoder_sd)
	ckpt_inp_dim, gat_hidden, gat_z = _infer_gat_hparams(graph_sd)
	if ckpt_inp_dim != inp_feature_dim:
		print(f"WARNING: graph feature dim {inp_feature_dim} != checkpoint {ckpt_inp_dim}")
	hidden_size = _cfg_int(index_data, "hidden_size", hidden_size)
	stack_width = _cfg_int(index_data, "stack_width", stack_width)
	z_dim = _cfg_int(index_data, "z_dim", z_dim)
	n_layers = _cfg_int(index_data, "n_layers", n_layers)
	if "bidir" in index_data:
		bidir = str(index_data["bidir"]).lower() in ("1", "true", "yes")

	embed = _cfg_int(index_data, "embed", _cfg_int(index_data, "max_length", 120))
	max_length = _cfg_int(index_data, "max_length", embed)
	batch_size = _cfg_int(index_data, "batch_size", 32)
	n_iters = _cfg_int(index_data, "n_iters", 100)
	sigma_val = _cfg_float(index_data, "sigma_val", 3)
	savecpt = _cfg_int(index_data, "savecpt", 1)
	stack_depth = _cfg_int(index_data, "stack_depth", 50)
	dropout = _cfg_float(index_data, "dropout", 0.2)
	train_batch_size = _cfg_int(index_data, "train_batch_size", 1)

	print(
		"Model architecture (from checkpoint): "
		f"hidden={hidden_size}, z={z_dim}, stack_width={stack_width}, "
		f"stack_depth={stack_depth}, n_layers={n_layers}, bidir={bidir}, "
		f"vocab={input_size}, graph_feat={inp_feature_dim}"
	)
	print("Single-pocket RL on receptor:", receptor)

	print("Building models...")
	prior_svae_encoder = VAE_Encoder(input_size, hidden_size, z_dim, output_size, train_batch_size,
		stack_depth, stack_width, dropout=dropout, n_layers=n_layers, bidir=bidir).to(device)
	prior_svae_decoder = VAE_Decoder(input_size, hidden_size, z_dim, output_size, train_batch_size,
		stack_depth, stack_width, dropout=dropout, n_layers=n_layers, bidir=bidir).to(device)
	prior_gvae_encoder = GAT_VAE_Encoder(inp_feature_dim, gat_hidden, gat_z, dropout=dropout).to(device)

	agent_svae_encoder = VAE_Encoder(input_size, hidden_size, z_dim, output_size, train_batch_size,
		stack_depth, stack_width, dropout=dropout, n_layers=n_layers, bidir=bidir).to(device)
	agent_svae_decoder = VAE_Decoder(input_size, hidden_size, z_dim, output_size, train_batch_size,
		stack_depth, stack_width, dropout=dropout, n_layers=n_layers, bidir=bidir).to(device)
	agent_gvae_encoder = GAT_VAE_Encoder(inp_feature_dim, gat_hidden, gat_z, dropout=dropout).to(device)

	prior_svae_encoder.load_state_dict(encoder_sd)
	prior_svae_decoder.load_state_dict(combined_state["decoder_state_dict"])
	prior_gvae_encoder.load_state_dict(graph_sd)
	agent_svae_encoder.load_state_dict(encoder_sd)
	agent_svae_decoder.load_state_dict(combined_state["decoder_state_dict"])
	agent_gvae_encoder.load_state_dict(graph_sd)

	predictor = None
	if os.path.isfile(predictor_cptfile):
		print("Loading predictor:", predictor_cptfile)
		with open(predictor_cptfile, "rb") as handle:
			predictor = pickle.load(handle)

	optimizer = optim.Adam(
		list(agent_svae_decoder.parameters()) + list(agent_gvae_encoder.parameters()),
		lr=float(index_data.get("lr", 1e-4)),
	)

	init_scorers()
	os.makedirs(savepath, exist_ok=True)

	cp = int(retraining_flag)
	print("Starting RL training: n_iters=", n_iters, "batch_size=", batch_size, "cp=", cp)
	rl_losses, avg_rewards = rltrainingloop(
		prot_A, prot_X,
		prior_svae_encoder, prior_svae_decoder, prior_gvae_encoder,
		agent_svae_encoder, agent_svae_decoder, agent_gvae_encoder,
		optimizer, predictor, savepath,
		char_to_int, int_to_char, max_length,
		batch_size, n_iters=n_iters, savecpt=savecpt, sigma_val=sigma_val, cp=cp,
	)
	print("RL training finished.")
	return rl_losses, avg_rewards


if __name__ == "__main__":
	run_training()
