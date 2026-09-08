"""
cvae_rl_integration.py — drop-in replacements for CVAE_RL training loop.

Copy the functions below into your main RL training script, OR import:

    from cvae_rl_integration import get_reward, policy_gradient

Changes vs original:
  - Removed docking (perform_docking / extract_docking_score)
  - Rewards: MW, QED, pIC50, solubility (mean of 4 terms)
  - predictor = get_predictor()  instead of  GBT = "yes"

Bottom-of-file change:
    # OLD:
    # GBT = "yes"
    # rl_losses, avg_rewards = rltrainingloop(..., GBT, docking_path, ...)

    # NEW:
    from RL import get_predictor
    predictor = get_predictor()
    rl_losses, avg_rewards = rltrainingloop(..., predictor, output_path, ...)
    # (docking_path no longer needed in policy_gradient)
"""

from __future__ import annotations

import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

_INFER_DIR = Path(__file__).resolve().parent
if str(_INFER_DIR) not in sys.path:
    sys.path.insert(0, str(_INFER_DIR))

from mw_scorer import calculateScore as mw_score
from pic50_scorer import calculateScore as pic50_score
from qed_scorer import calculateScore as qed_score
from RL import RewardPredictor, get_predictor
from sol_scorer import calculateScore as sol_score


# ---------------------------------------------------------------------------
# REWARD — replaces docking + SA + logP version
# ---------------------------------------------------------------------------

def get_reward(docking_outfile, smiles, predictor):
    """
    Combined reward: MW, QED, pIC50, solubility.

    docking_outfile: kept for CVAE_RL call signature — NOT USED.
    smiles:          generated SMILES string
    predictor:       RewardPredictor from get_predictor()
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0

    reward_mw = mw_score(mol)
    reward_qed = qed_score(mol)
    reward_pic50 = pic50_score(smiles, predictor.pic50)
    reward_sol = sol_score(smiles, predictor.sol)

    return float(np.mean([reward_mw, reward_qed, reward_pic50, reward_sol]))


# ---------------------------------------------------------------------------
# POLICY GRADIENT — docking step removed
# ---------------------------------------------------------------------------

def policy_gradient(
    X_adj,
    X_features,
    prior_svae_encoder,
    prior_svae_decoder,
    prior_gvae_encoder,
    agent_svae_encoder,
    agent_svae_decoder,
    agent_gvae_encoder,
    optimizer,
    predictor,
    output_path,
    filepath,
    char_to_int,
    int_to_char,
    max_len,
    nmols=256,
    sigma=3,
    sampling=None,
    tokenize=None,
    prior_likelihood=None,
):
    """
    Same as original policy_gradient but:
      - no perform_docking()
      - get_reward(None, trajectory, predictor) uses MW/QED/pIC50/sol

    Pass sampling, tokenize, prior_likelihood from your main script
    (they are defined in your CVAE_RL codebase).
    """
    if sampling is None or tokenize is None or prior_likelihood is None:
        raise ValueError("Pass sampling, tokenize, prior_likelihood from your main script")

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

    for _i in range(nmols):
        reward = 0
        while reward == 0:
            batch_gen_itercount += 1
            trajectory, traj_probs, top_indices = sampling(
                agent_svae_encoder,
                agent_svae_decoder,
                agent_gvae_encoder,
                X_adj,
                X_features,
                max_len,
                prime_str="!",
                end_token="E",
            )
            try:
                mol = Chem.MolFromSmiles(trajectory)
                if mol:
                    trajectory_tensor.append(trajectory)
                    traj_probs_tensor.append(traj_probs)
                    top_indices_tensor.append(top_indices)

                    # --- NEW: no docking; property-based reward only ---
                    reward = get_reward(None, trajectory, predictor)
                    print(trajectory, reward)
                    reward_tensor.append(reward)
                    graph_adj_tensor.append(X_adj)
                    graph_feat_tensor.append(X_features)
                else:
                    reward = 0
            except Exception:
                reward = 0
                continue

    if not reward_tensor:
        return 0.0, 0.0, batch_gen_itercount

    baseline_reward = sum(reward_tensor) / nmols

    for i in range(len(reward_tensor)):
        trajectory = trajectory_tensor[i]
        traj_probs = traj_probs_tensor[i]
        top_indices = top_indices_tensor[i]
        reward = reward_tensor[i]

        total_reward += reward
        prior_llh = prior_likelihood(
            prior_svae_encoder,
            prior_svae_decoder,
            prior_gvae_encoder,
            graph_adj_tensor[i],
            graph_feat_tensor[i],
            trajectory,
        )
        augmented_llh = prior_llh + reward * sigma

        agent_llh = 0.0
        for j in range(len(traj_probs)):
            output = traj_probs[j]
            log_probs = F.log_softmax(output, dim=-1)
            action_prob = log_probs[0, 0, top_indices[j]]
            agent_llh = agent_llh + action_prob

        returns = -((augmented_llh - agent_llh) ** 2)
        rl_loss = rl_loss + (-returns)

    rl_loss = rl_loss / len(reward_tensor)
    avg_reward = total_reward / len(reward_tensor)

    agent_svae_encoder.train()
    agent_svae_decoder.train()
    agent_gvae_encoder.train()
    prior_svae_encoder.train()
    prior_gvae_encoder.train()
    prior_svae_decoder.train()

    rl_loss.backward()
    nn.utils.clip_grad_norm_(
        list(agent_svae_decoder.parameters()) + list(agent_gvae_encoder.parameters()),
        3,
    )
    optimizer.step()

    return avg_reward, rl_loss.item(), batch_gen_itercount


# ---------------------------------------------------------------------------
# TRAINING LOOP — output_path instead of docking_path
# ---------------------------------------------------------------------------

def rltrainingloop(
    prot_A,
    prot_X,
    prior_svae_encoder,
    prior_svae_decoder,
    prior_gvae_encoder,
    agent_svae_encoder,
    agent_svae_decoder,
    agent_gvae_encoder,
    optimizer,
    predictor,
    output_path,
    char_to_int,
    int_to_char,
    max_length,
    batch_size,
    n_iters=100,
    savecpt=1,
    sigma_val=3,
    cp=0,
    savepath="",
    svae_gvae_combined_cptfile=None,
    device=None,
    sampling=None,
    tokenize=None,
    prior_likelihood=None,
    prior_smiles_encoder=None,
    prior_smiles_decoder=None,
    agent_smiles_encoder=None,
    agent_smiles_decoder=None,
    prior_gcn_encoder=None,
    agent_gcn_encoder=None,
):
    total_rl_loss = []
    avg_rl_reward = []

    pg_kwargs = {
        "sampling": sampling,
        "tokenize": tokenize,
        "prior_likelihood": prior_likelihood,
    }

    if cp == 0:
        for iteration in range(n_iters):
            filepath = output_path
            dirpath = os.path.join(output_path, "RL_iter_" + str(iteration))
            os.mkdir(dirpath)

            avg_reward, rl_loss, batchitercount = policy_gradient(
                prot_A,
                prot_X,
                prior_svae_encoder,
                prior_svae_decoder,
                prior_gvae_encoder,
                agent_svae_encoder,
                agent_svae_decoder,
                agent_gvae_encoder,
                optimizer,
                predictor,
                dirpath,
                filepath,
                char_to_int,
                int_to_char,
                max_length,
                nmols=batch_size,
                sigma=sigma_val,
                **pg_kwargs,
            )
            print(
                "Iter:",
                iteration + 1,
                "Average reward:",
                avg_reward,
                "RL loss:",
                rl_loss,
                "Sampling iterations:",
                batchitercount,
            )
            total_rl_loss.append(rl_loss)
            avg_rl_reward.append(avg_reward)
            if (iteration + 1) % savecpt == 0:
                path = savepath + "RL_Weights_epoch_" + str(iteration + 1) + ".cpt"
                state = {
                    "epoch": iteration + 1,
                    "encoder_state_dict": agent_svae_encoder.state_dict(),
                    "decoder_state_dict": agent_svae_decoder.state_dict(),
                    "graph_encoder_state_dict": agent_gvae_encoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rl_loss": rl_loss,
                    "avg_reward": avg_reward,
                    "batchitercount": batchitercount,
                }
                torch.save(state, path)
            else:
                shutil.rmtree(filepath + "RL_iter_" + str(iteration))

    else:
        combined_state_old = torch.load(svae_gvae_combined_cptfile, map_location=device)
        prior_smiles_encoder.load_state_dict(combined_state_old["encoder_state_dict"])
        prior_smiles_decoder.load_state_dict(combined_state_old["decoder_state_dict"])
        agent_smiles_encoder.load_state_dict(combined_state_old["encoder_state_dict"])
        agent_smiles_decoder.load_state_dict(combined_state_old["decoder_state_dict"])
        prior_gcn_encoder.load_state_dict(combined_state_old["graph_encoder_state_dict"])
        agent_gcn_encoder.load_state_dict(combined_state_old["graph_encoder_state_dict"])
        optimizer.load_state_dict(combined_state_old["optimizer"])

        print("Old model weights loaded...")
        print("Training continued...")

        iteration_old = combined_state_old["epoch"]
        for iteration in range(iteration_old + 1, n_iters):
            filepath = output_path
            dirpath = os.path.join(output_path, "RL_iter_" + str(iteration))
            os.mkdir(dirpath)

            avg_reward, rl_loss, batchitercount = policy_gradient(
                prot_A,
                prot_X,
                prior_svae_encoder,
                prior_svae_decoder,
                prior_gvae_encoder,
                agent_svae_encoder,
                agent_svae_decoder,
                agent_gvae_encoder,
                optimizer,
                predictor,
                dirpath,
                filepath,
                char_to_int,
                int_to_char,
                max_length,
                nmols=batch_size,
                sigma=sigma_val,
                **pg_kwargs,
            )
            print(
                "Iter:",
                iteration + 1,
                "Average reward:",
                avg_reward,
                "RL loss:",
                rl_loss,
                "Sampling iterations:",
                batchitercount,
            )
            total_rl_loss.append(rl_loss)
            avg_rl_reward.append(avg_reward)
            if (iteration + 1) % savecpt == 0:
                path = savepath + "RL_Weights_epoch_" + str(iteration + 1) + ".cpt"
                state = {
                    "epoch": iteration + 1,
                    "encoder_state_dict": agent_svae_encoder.state_dict(),
                    "decoder_state_dict": agent_svae_decoder.state_dict(),
                    "graph_encoder_state_dict": agent_gvae_encoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rl_loss": rl_loss,
                    "avg_reward": avg_reward,
                    "batchitercount": batchitercount,
                }
                torch.save(state, path)
            else:
                shutil.rmtree(filepath + "RL_iter_" + str(iteration))

    return total_rl_loss, avg_rl_reward
