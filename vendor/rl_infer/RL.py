"""
RL.py — get_reward, policy_gradient, rltrainingloop

Same 3-function structure as CVAE_RL. Only get_reward extended with
MW, QED, pIC50, solubility (geometric mean of all 7 terms).

Requires in your main script (globals used by policy_gradient / rltrainingloop):
  perform_docking, sampling, tokenize, prior_likelihood, savepath,
  svae_gvae_combined_cptfile, device, prior_smiles_encoder, ...
"""

from __future__ import annotations

import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors

_INFER_DIR = Path(__file__).resolve().parent
if str(_INFER_DIR) not in sys.path:
    sys.path.insert(0, str(_INFER_DIR))

from docking_scorer import extract_docking_score
from mw_scorer import calculateScore as mw_score
from pic50_scorer import Pic50Predictor, calculateScore as pic50_score
from qed_scorer import calculateScore as qed_score
from sascorer import calculateScore as sa_raw_score
from sol_scorer import SolPredictor, calculateScore as sol_score


@dataclass
class RewardPredictor:
    pic50: Pic50Predictor
    sol: SolPredictor


_predictor: RewardPredictor | None = None


def get_predictor() -> RewardPredictor:
    global _predictor
    if _predictor is None:
        _predictor = RewardPredictor(pic50=Pic50Predictor(), sol=SolPredictor())
    return _predictor


def geometric_mean_rewards(rewards: list[float]) -> float:
    if not rewards:
        return 0.0
    vals = np.array(rewards, dtype=np.float64)
    if not np.all(np.isfinite(vals)) or np.any(vals <= 0):
        return 0.0
    return float(np.exp(np.mean(np.log(vals))))


# ---------------------------------------------------------------------------
# get_reward — same signature; original + new terms; geometric mean
# ---------------------------------------------------------------------------

def get_reward(docking_outfile, smiles, predictor):
    reward1 = 0
    dockscore = extract_docking_score(docking_outfile)
    reward1 = math.exp(-dockscore / 3.0)

    rdkitmol = Chem.MolFromSmiles(smiles)
    if rdkitmol is None:
        return 0.0

    reward2 = 0
    sas = sa_raw_score(rdkitmol)
    reward2 = math.exp(sas / 3.0)

    reward3 = 0
    clogp = Descriptors.MolLogP(rdkitmol)
    if clogp >= -1 and clogp <= 3:
        reward3 = 11
    else:
        reward3 = 1

    reward4 = mw_score(rdkitmol)
    reward5 = qed_score(rdkitmol)
    reward6 = pic50_score(smiles, predictor.pic50)
    reward7 = sol_score(smiles, predictor.sol)

    reward = geometric_mean_rewards([reward1, reward2, reward3, reward4, reward5, reward6, reward7])
    return reward


# ---------------------------------------------------------------------------
# policy_gradient — unchanged structure
# ---------------------------------------------------------------------------

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

    for i in range(nmols):
        reward = 0
        while reward == 0:
            batch_gen_itercount = batch_gen_itercount + 1
            trajectory, traj_probs, top_indices = sampling(agent_svae_encoder, agent_svae_decoder, agent_gvae_encoder, X_adj, X_features, max_len, prime_str='!', end_token='E')
            try:
                mol = Chem.MolFromSmiles(trajectory)
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
                    reward = 0
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


# ---------------------------------------------------------------------------
# rltrainingloop — unchanged structure
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Optional: batch breakdown (used by infer.py only)
# ---------------------------------------------------------------------------

def get_reward_breakdown(docking_outfile, smiles, predictor=None):
    if predictor is None:
        predictor = get_predictor()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"reward": 0.0}

    dockscore = float("nan")
    reward1 = 0.0
    if docking_outfile:
        try:
            dockscore = extract_docking_score(docking_outfile)
            reward1 = math.exp(-dockscore / 3.0)
        except Exception:
            reward1 = 0.0

    sas = sa_raw_score(mol)
    reward2 = math.exp(sas / 3.0)
    clogp = Descriptors.MolLogP(mol)
    reward3 = 11.0 if -1 <= clogp <= 3 else 1.0
    reward4 = mw_score(mol)
    reward5 = qed_score(mol)
    reward6 = pic50_score(smiles, predictor.pic50)
    reward7 = sol_score(smiles, predictor.sol)

    terms = [reward1, reward2, reward3, reward4, reward5, reward6, reward7]
    if not docking_outfile:
        terms = terms[1:]

    return {
        "reward": geometric_mean_rewards(terms),
        "reward_docking": reward1,
        "reward_sa": reward2,
        "reward_logp": reward3,
        "reward_mw": reward4,
        "reward_qed": reward5,
        "reward_pic50": reward6,
        "reward_sol": reward7,
        "docking_score": dockscore,
        "sa": sas,
        "logp": clogp,
        "pic50": predictor.pic50.predict(smiles),
        "solubility": predictor.sol.predict(smiles),
    }
