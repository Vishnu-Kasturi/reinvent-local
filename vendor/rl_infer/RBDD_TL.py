#Script to create the conditional molecule generator and re-train it with SMILES-binding site graph pairs

import sys
import os
import numpy as np
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss
import time
import utils

from utils import Dataset
from smiles_vae_model import Stack_GRU, VAE_Encoder, VAE_Decoder
from gat_vae_model import GraphAttentionLayer, GAT_VAE_Encoder

from torch import optim
from torch.utils import data
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from scipy.special import expit
from io import open

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device available:",device)
cpu = torch.device("cpu")

#Command-line arguments - File names
indexfile = sys.argv[1]  #Input dataset of SMILES-binding site graph pairs
char_to_int_file = sys.argv[2]  #Character to integer mapping
int_to_char_file = sys.argv[3]  #Integer to character mapping
graphpath = sys.argv[4]  #Path to the graph dataset files
svaecptfile = sys.argv[5]  #Path to the pre-trained SMILES VAE model checkpoint file
gvaecptfile = sys.argv[6]  #Path to the pre-trained GAT VAE model checkpoint file
savepath = sys.argv[7]  #Path to save the checkpoint files of the combined model


#Hyperparameter details - Architecture of the CVAE model
latent_dim = 256
gru_dim = 1024
nbatches_train = 2000
nbatches_val = 2000
stack_width = 256
batch_size = 1
embed = 160
kl_growth_rate = 0.05
feature_dim = 9
gcn_latent_dim = 256
gcn_hidden_dim = 128

#---------------------------------------------------------SMILES VAE-SPECIFIC FUNCTIONS----------------------------------------------------
#Used to enable backpropagation through a sampling process - Approximating the latent manifold using a mixture of Gaussians
def reparameterization_trick(mu, sigma):
	std = torch.exp(0.5 * sigma)
	eps = torch.randn_like(std)
	latent_z = eps.mul(std).add_(mu)
	return latent_z

#KL divergence calculation for the VAE joint loss
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
	mu = mu.to(device)
	logvar = logvar.to(device)
	kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	return kld
#-------------------------------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------GAT VAE-SPECIFIC FUNCTIONS----------------------------------------------------
#Used to enable backpropagation through a sampling process - Approximating the latent manifold using a mixture of Gaussians
def reparameterization_trick_graph(mu, sigma):
	std = torch.exp(sigma)
	eps = torch.randn_like(std)
	latent_z = eps.mul(std).add_(mu)
	return latent_z

#GAT VAE joint loss function - Binary cross entropy loss with logits + KL-divergence
def joint_loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
	mu = mu.double().to(device)
	logvar = logvar.double().to(device)
	preds = preds.double().to(device)
	labels = labels.double().to(device)
	cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

	KLD = (-0.5 / n_nodes) * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
	final_loss = 0.0
	final_loss = cost + KLD
	return final_loss

#ROC and AP score calculation function
def get_roc_score(adj_reconstructed, adj_original, edges_positive, edges_negative):
	#Function to return the sigmoid of a given scalar/vector
	def sigmoid(x):
		return 1 / (1 + expit(-x))  #Use expit() instead of math.exp() or np.exp() to avoid overflow errors

	#Predict on test set of edges
	preds = []
	pos = []
	for edge in edges_positive:  #edges_positive and edges_negative should be a 2D array with the source and target edge positions
		preds.append(sigmoid(adj_reconstructed[edge[0], edge[1]]))
		pos.append(adj_original[edge[0], edge[1]])

	preds_neg = []
	neg = []
	for edge in edges_negative:
		preds_neg.append(sigmoid(adj_reconstructed[edge[0], edge[1]]))
		neg.append(adj_original[edge[0], edge[1]])

	preds_all = np.hstack([preds, preds_neg])
	labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
	roc_score = roc_auc_score(labels_all, preds_all)
	ap_score = average_precision_score(labels_all, preds_all)

	return roc_score, ap_score
#--------------------------------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------TRAINING HELPER FUNCTION---------------------------------------------------
def train(dataset, svae_encoder, svae_decoder, gvae_encoder, optimizer, criterion, max_length, batch_size, vocab_size, ntrain, step, kl_growth):
	svae_encoder.train()
	svae_decoder.train()
	gvae_encoder.train()

	start_time = time.time()
	batches = ntrain
	total_loss = 0.
	total_acc = 0.
	minibatch_size = 256
	niter = batches/minibatch_size

	for iteration in range(0, round(niter)):
		optimizer.zero_grad()
		joint_loss = torch.zeros(1, 1).to(device, dtype=torch.float64)
		batchcount = 0

		for inputbatch, targetbatch, X_adj, X_features, Y_adj, Y_features, pos_weight, norm in dataset:
			input_tensor = inputbatch.to(device)
			target_tensor = targetbatch.to(device)
			X_adj = X_adj.to(device)
			Y_adj = Y_adj.to(device)
			X_features = X_features.to(device)
			Y_features = Y_features.to(device)

			input_length = input_tensor.size(1)
			target_length = target_tensor.size(1)

			prediction = []
			g_t = []

			train_loss = torch.zeros(1, 1).to(device)
			train_acc = torch.zeros(1, 1).to(device)

			hidden_forward = svae_encoder.forward_stackgru.initHidden()
			stack_forward = svae_encoder.forward_stackgru.initStack()
			decoder_stack = svae_decoder.decoder_stackgru.initStack()

			graph_mean, graph_logvar = gvae_encoder(X_features, X_adj)
			graph_latent_z = reparameterization_trick_graph(graph_mean, graph_logvar)
			graph_latent_z = torch.mean(graph_latent_z, dim=0)
			graph_latent_z = graph_latent_z.unsqueeze(0)
			graph_latent_z = graph_latent_z.unsqueeze(0)

			if(svae_encoder.numdir==1):
				#Encoder training
				for k in range(input_length):
					hidden_forward = svae_encoder.forward_unidir(input_tensor[:,k], hidden_forward, stack_forward)
					break  #Stopping the encoder once the "!" character has been passed to through

				#Latent vector generation
				smiles_mean, smiles_logvar = svae_encoder.post_gru_reshape_function_unidir(hidden_forward)
				smiles_latent_z = reparameterization_trick(smiles_mean, smiles_logvar)
				smiles_latent_z = smiles_latent_z.unsqueeze(0)
				latent_z = smiles_latent_z + graph_latent_z  #Adding both latent vectors to get the joint latent vector

				decoder_hidden = svae_decoder.latent_vector_to_hidden(latent_z)

				#Decoder training
				for k in range(input_length):
					output, decoder_hidden, decoder_stack = svae_decoder(input_tensor[:,k], decoder_hidden, decoder_stack)
					train_loss += criterion(output.view(-1,vocab_size).to(device),target_tensor[:,k].view(-1).to(device))

					for b in range(batch_size):
						prediction.append(np.argmax(output[0,b,:].cpu().detach().numpy()))
						g_t.append(target_tensor[b,k].cpu().detach().numpy())

					joint_loss += train_loss
					train_acc = accuracy_score(prediction,g_t)
					total_acc += train_acc.item()

			else:
				hidden_backward = svae_encoder.backward_stackgru.initHidden()
				stack_backward = svae_encoder.backward_stackgru.initStack()

				#Encoder_training
				backward_index = input_length - 1
				for k in range(input_length):
					hidden_forward, hidden_backward = svae_encoder.forward_bidir(input_tensor[:,k], input_tensor[:,backward_index], hidden_forward, stack_forward, hidden_backward, stack_backward)
					backward_index = backward_index - 1
					break  #Stopping the encoder once the "!" character has been passed to through

				#Latent vector generation
				smiles_mean, smiles_logvar = svae_encoder.post_gru_reshape_function_bidir(hidden_forward, hidden_backward)
				smiles_latent_z = reparameterization_trick(smiles_mean, smiles_logvar)
				smiles_latent_z = smiles_latent_z.unsqueeze(0)
				latent_z = smiles_latent_z + graph_latent_z  #Adding both latent vectors to get the joint latent vector
			
				decoder_hidden = svae_decoder.latent_vector_to_hidden(latent_z)

				#Decoder_training
				for k in range(input_length):
					output, decoder_hidden, decoder_stack = svae_decoder(input_tensor[:,k], decoder_hidden, decoder_stack)
					train_loss += criterion(output.view(-1,vocab_size).to(device),target_tensor[:,k].view(-1).to(device))

					for b in range(batch_size):
						prediction.append(np.argmax(output[0,b,:].cpu().detach().numpy()))
						g_t.append(target_tensor[b,k].cpu().detach().numpy())

					joint_loss += train_loss
					train_acc = accuracy_score(prediction,g_t)
					total_acc += train_acc.item()

			#Stop if 1 minibatch is completed
			batchcount = batchcount + 1
			if(batchcount==minibatch_size):
				break

		joint_loss = joint_loss/minibatch_size
		joint_loss.backward()
		optimizer.step()
		total_loss += joint_loss.item()
		total_acc = total_acc/minibatch_size

	total_loss = total_loss/niter
	total_acc = total_acc/niter
	end_time = time.time()
	return end_time-start_time, total_loss, total_acc

#---------------------------------------------------------VALIDATION HELPER FUNCTION---------------------------------------------------
def validation(dataset, svae_encoder, svae_decoder, gvae_encoder, max_length, criterion, batch_size, vocab_size, ntrain, step, kl_growth):
	with torch.no_grad():
		svae_encoder.eval()
		svae_decoder.eval()
		gvae_encoder.eval()

		start_time = time.time()
		batches = ntrain
		total_loss = 0.
		total_acc = 0.
		minibatch_size = 256
		niter = batches/minibatch_size

		for iteration in range(0, round(niter)):
			joint_loss = torch.zeros(1, 1).to(device, dtype=torch.float64)
			batchcount = 0

			for inputbatch, targetbatch, X_adj, X_features, Y_adj, Y_features, pos_weight, norm in dataset:
				input_tensor = inputbatch.to(device)
				target_tensor = targetbatch.to(device)
				X_adj = X_adj.to(device)
				Y_adj = Y_adj.to(device)
				X_features = X_features.to(device)
				Y_features = Y_features.to(device)

				input_length = input_tensor.size(1)
				target_length = target_tensor.size(1)

				prediction = []
				g_t = []

				val_loss = torch.zeros(1, 1).to(device)
				val_acc = torch.zeros(1, 1).to(device)

				hidden_forward = svae_encoder.forward_stackgru.initHidden()
				stack_forward = svae_encoder.forward_stackgru.initStack()
				decoder_stack = svae_decoder.decoder_stackgru.initStack()

				graph_mean, graph_logvar = gvae_encoder(X_features, X_adj)
				graph_latent_z = reparameterization_trick_graph(graph_mean, graph_logvar)
				graph_latent_z = torch.mean(graph_latent_z, dim=0)
				graph_latent_z = graph_latent_z.unsqueeze(0)
				graph_latent_z = graph_latent_z.unsqueeze(0)

				if(svae_encoder.numdir==1):
					#Encoder training
					for k in range(input_length):
						hidden_forward = svae_encoder.forward_unidir(input_tensor[:,k], hidden_forward, stack_forward)
						break  #Stopping the encoder once the "!" character has been passed through

					#Latent vector generation
					smiles_mean, smiles_logvar = svae_encoder.post_gru_reshape_function_unidir(hidden_forward)
					smiles_latent_z = reparameterization_trick(smiles_mean, smiles_logvar)
					smiles_latent_z = smiles_latent_z.unsqueeze(0)
					latent_z = smiles_latent_z + graph_latent_z

					decoder_hidden = svae_decoder.latent_vector_to_hidden(latent_z)

					#Decoder training
					for k in range(input_length):
						output, decoder_hidden, decoder_stack = svae_decoder(input_tensor[:,k], decoder_hidden, decoder_stack)
						val_loss += criterion(output.view(-1,vocab_size).to(device),target_tensor[:,k].view(-1).to(device))

						for b in range(batch_size):
							prediction.append(np.argmax(output[0,b,:].cpu().detach().numpy()))
							g_t.append(target_tensor[b,k].cpu().detach().numpy())

						joint_loss += val_loss
						val_acc = accuracy_score(prediction,g_t)
						total_acc += val_acc.item()

				else:
					hidden_backward = svae_encoder.backward_stackgru.initHidden()
					stack_backward = svae_encoder.backward_stackgru.initStack()

					#Encoder_training
					backward_index = input_length - 1
					for k in range(input_length):
						hidden_forward, hidden_backward = svae_encoder.forward_bidir(input_tensor[:,k], input_tensor[:,backward_index], hidden_forward, stack_forward, hidden_backward, stack_backward)
						backward_index = backward_index - 1
						break

					#Latent vector generation
					smiles_mean, smiles_logvar = svae_encoder.post_gru_reshape_function_bidir(hidden_forward, hidden_backward)
					smiles_latent_z = reparameterization_trick(smiles_mean, smiles_logvar)
					smiles_latent_z = smiles_latent_z.unsqueeze(0)
					latent_z = smiles_latent_z + graph_latent_z

					decoder_hidden = svae_decoder.latent_vector_to_hidden(latent_z)

					#Decoder_training
					for k in range(input_length):
						output, decoder_hidden, decoder_stack = svae_decoder(input_tensor[:,k], decoder_hidden, decoder_stack)
						val_loss += criterion(output.view(-1,vocab_size).to(device),target_tensor[:,k].view(-1).to(device))

						for b in range(batch_size):
							prediction.append(np.argmax(output[0,b,:].cpu().detach().numpy()))
							g_t.append(target_tensor[b,k].cpu().detach().numpy())

						joint_loss += val_loss
						val_acc = accuracy_score(prediction,g_t)
						total_acc += val_acc.item()

				#Stop if 1 minibatch is completed
				batchcount = batchcount + 1
				if(batchcount==minibatch_size):
					break

			joint_loss = joint_loss/minibatch_size
			total_loss += joint_loss.item()
			total_acc = total_acc/minibatch_size

		total_loss = total_loss/niter
		total_acc = total_acc/niter
		end_time = time.time()
	return end_time-start_time, total_loss, total_acc


#-------------------------------------------------------------TRAINING LOOP-----------------------------------------------------------
def trainingloop(trainset, valset, max_length, svae_encoder, svae_decoder, gvae_encoder, n_iters, vocab_size, learning_rate=0.0001, batch_size=256, savecpt=10, ntrain=200, nval=200, kl_annealing=0.004, cp=0):
	start = time.time()

	train_losses = []
	validation_losses = []

	optimizer = optim.Adam(list(gvae_encoder.parameters()) + list(svae_decoder.parameters()), lr=learning_rate, amsgrad=True)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',verbose=True,patience=5)
	criterion = nn.CrossEntropyLoss()

	if(cp==1):
		checkpoint = torch.load(str(savepath)+'/Weights_epoch_45.cpt')
		svae_encoder.load_state_dict(checkpoint['encoder_state_dict'])
		svae_decoder.load_state_dict(checkpoint['decoder_state_dict'])
		gvae_encoder.load_state_dict(checkpoint['graph_encoder_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		previous_epoch = checkpoint['epoch']
		print('After loading the weights')

		for epoch in range(n_iters):
			time_taken, train_loss, train_acc = train(trainset, svae_encoder, svae_decoder, gvae_encoder, optimizer, criterion, max_length, batch_size, vocab_size, ntrain, epoch, kl_annealing)
			time_taken, val_loss, val_acc = validation(valset, svae_encoder, svae_decoder, gvae_encoder, max_length, criterion, batch_size, vocab_size, nval, epoch, kl_annealing)
			train_losses.append(train_loss)
			validation_losses.append(val_loss)
			print("Epoch_no: ",(previous_epoch+epoch+1) ,"Training loss: ",train_loss, "Training acc: ",train_acc, "Validation loss: ",val_loss, "Validation acc: ",val_acc)
			if(((previous_epoch+epoch+1)%savecpt) == 0):
				path = str(savepath)+"/Weights_epoch_"+str(previous_epoch+epoch+1)+".cpt"
				state = {'epoch': previous_epoch+epoch+1, 'encoder_state_dict': svae_encoder.state_dict(), 'decoder_state_dict': svae_decoder.state_dict(), 'graph_encoder_state_dict': gvae_encoder.state_dict(),'optimizer': optimizer.state_dict(), 'train_loss':  train_loss, 'val_loss': val_loss, 'train_acc': train_acc, 'val_acc': val_acc}
				torch.save(state, path)
			scheduler.step(val_loss)


	for epoch in range(n_iters):
		time_taken, train_loss, train_acc = train(trainset, svae_encoder, svae_decoder, gvae_encoder, optimizer, criterion, max_length, batch_size, vocab_size, ntrain, epoch, kl_annealing)
		time_taken, val_loss, val_acc = validation(valset, svae_encoder, svae_decoder, gvae_encoder, max_length, criterion, batch_size, vocab_size, nval, epoch, kl_annealing)
		train_losses.append(train_loss)
		validation_losses.append(val_loss)
		print("Epoch_no: ",epoch ,"Training loss: ",train_loss, "Training acc: ",train_acc, "Validation loss: ",val_loss, "Validation acc: ",val_acc)
		if(((epoch+1)%savecpt) == 0):
			path = str(savepath)+"/Weights_epoch_"+str(epoch+1)+".cpt"
			state = {'epoch': epoch+1, 'encoder_state_dict': svae_encoder.state_dict(), 'decoder_state_dict': svae_decoder.state_dict(), 'graph_encoder_state_dict': gvae_encoder.state_dict(),'optimizer': optimizer.state_dict(), 'train_loss':  train_loss, 'val_loss': val_loss, 'train_acc': train_acc, 'val_acc': val_acc}
			torch.save(state, path)

	return train_losses, validation_losses


#-------------------------------------------------------LOSS PLOTTING FUNCTION-----------------------------------------------------
def plotlossfunc(train_losses,val_losses):
	plt.plot(train_losses)
	plt.plot(val_losses)
	plt.xlabel('Epoch')
	plt.ylabel('Loss (Cross-entropy)')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('Loss_history_v1.png', bbox_inches='tight')


#------------------------------------------------------FUNCTION CALL STATEMENTS----------------------------------------------------
#Read the list of PDB IDs from the file into a list
indexlist = []
smilesmap = {}
with open(indexfile) as molecules:
	molreader = csv.reader(molecules, delimiter='\t', quotechar='"') #Creates a file handle
	for molecule in molreader:
		smiles = ''.join(molecule[0]).strip()
		file_id = ''.join(molecule[1])
		indexlist.append(file_id)
		smilesmap[file_id] = smiles

#Split the dataset indexes into train and test set indexes
index_train, index_test = train_test_split(indexlist, random_state=42)

#Tokenize the SMILES to pass to the DataLoader
tokenized_smiles_dict, charset, char_to_int, int_to_char = utils.prepare_dicts_and_dictfiles(indexlist, smilesmap, char_to_int_file, int_to_char_file)
print("SMILES_vocabulary:", str(charset))

#Vocabulary size = No. of unique elements in the SMILES vocabulary
vocab_size = len(charset)

#Depth of stack - Need not exceed the maximum length of SMILES in the input dataset
stack_depth = embed

#Batching using the PyTorch DataLoader
train_obj = Dataset(index_train, tokenized_smiles_dict, graphpath)
test_obj = Dataset(index_test, tokenized_smiles_dict, graphpath)
train_batches = data.DataLoader(train_obj, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
test_batches = data.DataLoader(test_obj, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=1)

print("Minibatches prepared...")

#Model instance definition
smiles_encoder = VAE_Encoder(vocab_size, gru_dim, latent_dim, vocab_size, batch_size, stack_depth, stack_width, dropout=0.2, n_layers=2, bidir=True).to(device)
smiles_decoder = VAE_Decoder(vocab_size, gru_dim, latent_dim, vocab_size, batch_size, stack_depth, stack_width, dropout=0.2, n_layers=2, bidir=False).to(device)
gcn_encoder = GAT_VAE_Encoder(feature_dim, gcn_hidden_dim, gcn_latent_dim, dropout=0.2).to(device)

#Loading the best model weights on the model objects
svae_state_old = torch.load(svaecptfile, map_location=device)
smiles_encoder.load_state_dict(svae_state_old['encoder_state_dict'])
smiles_decoder.load_state_dict(svae_state_old['decoder_state_dict'])

gvae_state_old = torch.load(gvaecptfile, map_location=device)
gcn_encoder.load_state_dict(gvae_state_old['encoder_state_dict'])

print("Model weights loaded...")

#Training step
print("Training begins now...")
train_loss, val_loss = trainingloop(train_batches, test_batches, embed, smiles_encoder, smiles_decoder, gcn_encoder, 500, vocab_size,  learning_rate=0.0005, batch_size=batch_size, savecpt=5, ntrain=nbatches_train, nval=nbatches_val, kl_annealing=kl_growth_rate, cp=0)
#plotlossfunc(train_loss, val_loss)


