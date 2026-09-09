import math
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

from sascorer import readFragmentScores
from pic50_scorer import readModel as readPic50Model, calculateScore as calculatePic50
from sol_scorer import readModel as readSolModel, calculateScore as calculateSolubility
from reinvent_transforms import (
    REWARD_WEIGHTS,
    transform_docking,
    transform_pic50,
    transform_solubility,
    transform_tyrosine,
    weighted_geometric_mean,
)
from dock_prolif_backend import SingleMoleculeCache


def init_scorers():
    """Call once before RL training (same as readFragmentScores() in original script)."""
    readFragmentScores()
    readPic50Model()
    readSolModel()


#def get_reward(ecif_input, smiles, predictor): #Specific to logP - Based on Popova et al
def get_reward(dock_result, smiles, predictor):
    rdkitmol = Chem.MolFromSmiles(smiles)
    if rdkitmol is None or dock_result is None or not dock_result.docking_ok:
        return 0.0

    reward_docking = transform_docking(dock_result.affinity)
    reward_tyr = transform_tyrosine(dock_result.tyr_pi_stacking_count)

    pic50 = calculatePic50(smiles)
    if not math.isfinite(pic50):
        return 0.0
    reward_pic50 = transform_pic50(pic50)

    sol = calculateSolubility(smiles)
    if not math.isfinite(sol):
        return 0.0
    reward_sol = transform_solubility(sol)

    return weighted_geometric_mean([
        (reward_sol, REWARD_WEIGHTS["solubility"]),
        (reward_pic50, REWARD_WEIGHTS["pic50"]),
        (reward_tyr, REWARD_WEIGHTS["tyrosine"]),
        (reward_docking, REWARD_WEIGHTS["docking"]),
    ])


#-------------------------------------------------------RL POLICY GRADIENT UPDATE FUNCTION------------------------------------------
def policy_gradient(X_adj, X_features, prior_svae_encoder, prior_svae_decoder, prior_gvae_encoder, agent_svae_encoder, agent_svae_decoder, agent_gvae_encoder, optimizer, predictor, output_path, filepath, char_to_int, int_to_char, max_len, nmols=256, sigma=3):
    rl_loss = 0
    optimizer.zero_grad()
    total_reward = 0
    avg_reward = 0
    batch_gen_itercount = 0
    SingleMoleculeCache.clear()

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

                    dock_result = SingleMoleculeCache.get_or_run(trajectory, batch_gen_itercount, output_path)
                    reward = get_reward(dock_result, trajectory, predictor)
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
