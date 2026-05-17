"""
QSAR 5 v3: High-Performance Tabular Neural Network
===================================================
Improvements:
  - Enriched Features: PhysChem(12), MACCS(167), ECFP4(2048), ECFP6(2048), RDKitTopo(2048)
  - Zone-Aware Weighting: low=4.0, bulk=1.0, high=3.0 to focus on extremes
  - Architecture: Residual Multi-Layer Perceptron (ResMLP) with Grouped Projections
  - Optimization: OneCycleLR scheduler for superior convergence

Usage:
  python qsar5.py --train data_splits_random/train.csv --val data_splits_random/val.csv --test data_splits_random/test.csv
"""

import os, math, argparse, json, warnings, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem, RDKFingerprint
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PHYSCHEM_COLS = [
    "mw", "exact_mw", "logp", "hbd", "hba", "tpsa",
    "rot_bonds", "rings", "arom_rings",
    "heavy_atoms", "frac_csp3", "stereo",
]
ZONE_WEIGHTS = {"low": 4.0, "bulk": 1.0, "high": 3.0}

# ══════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════
class QsarRichDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.smiles = df["canon_smiles"].tolist()
        self.labels = torch.tensor(df["pic50"].values, dtype=torch.float32)
        
        # Ensure 12 PhysChem features
        if "exact_mw" not in df.columns:
            df["exact_mw"] = df["mw"]
        self.physchem = torch.tensor(df[PHYSCHEM_COLS].values, dtype=torch.float32)
        
        # Zone weights
        if "zone" in df.columns:
            self.weights = torch.tensor([ZONE_WEIGHTS.get(z, 1.0) for z in df["zone"]], dtype=torch.float32)
        else:
            self.weights = torch.ones(len(df), dtype=torch.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        mol = Chem.MolFromSmiles(str(smi))
        
        if mol is None:
            maccs = np.zeros(167, 32); e4 = np.zeros(2048, 32); e6 = np.zeros(2048, 32); rdk = np.zeros(2048, 32)
        else:
            maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
            e4 = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048), dtype=np.float32)
            e6 = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 3, 2048), dtype=np.float32)
            rdk = np.array(RDKFingerprint(mol, fpSize=2048), dtype=np.float32)
            
        return {
            "pc": self.physchem[idx],
            "maccs": torch.tensor(maccs),
            "e4": torch.tensor(e4),
            "e6": torch.tensor(e6),
            "rdk": torch.tensor(rdk),
            "label": self.labels[idx],
            "weight": self.weights[idx]
        }

# ══════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════
class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(x + self.block(x))

class QsarResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Projections
        self.pc_proj = nn.Linear(12, 128)
        self.maccs_proj = nn.Linear(167, 128)
        self.e4_proj = nn.Linear(2048, 256)
        self.e6_proj = nn.Linear(2048, 256)
        self.rdk_proj = nn.Linear(2048, 256)
        
        combined_dim = 128 + 128 + 256 + 256 + 256 # 1024
        self.ln_in = nn.LayerNorm(combined_dim)
        
        self.res_layers = nn.Sequential(
            nn.Linear(combined_dim, 512),
            ResBlock(512),
            ResBlock(512),
            nn.Linear(512, 256),
            ResBlock(256),
            nn.Linear(256, 1)
        )

    def forward(self, pc, maccs, e4, e6, rdk):
        x = torch.cat([
            self.pc_proj(pc), self.maccs_proj(maccs), 
            self.e4_proj(e4), self.e6_proj(e6), self.rdk_proj(rdk)
        ], dim=1)
        x = self.ln_in(x)
        return self.res_layers(x).squeeze(-1)

# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for b in loader:
            yhat = model(b["pc"].to(device), b["maccs"].to(device), b["e4"].to(device), b["e6"].to(device), b["rdk"].to(device))
            preds.extend(yhat.cpu().numpy())
            targets.extend(b["label"].numpy())
    return r2_score(targets, preds), math.sqrt(mean_squared_error(targets, preds))

def main(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nDevice: {device}")

    train_df, val_df, test_df = pd.read_csv(args.train), pd.read_csv(args.val), pd.read_csv(args.test)
    train_loader = DataLoader(QsarRichDataset(train_df), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(QsarRichDataset(val_df),   batch_size=args.batch_size)
    test_loader  = DataLoader(QsarRichDataset(test_df),  batch_size=args.batch_size)

    model = QsarResNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)
    
    print(f"\nTraining for {args.epochs} epochs with ResMLP …")
    best_r2 = -float("inf")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for b in train_loader:
            pc, ma, e4, e6, rdk, y, w = b["pc"].to(device), b["maccs"].to(device), b["e4"].to(device), b["e6"].to(device), b["rdk"].to(device), b["label"].to(device), b["weight"].to(device)
            optimizer.zero_grad()
            yhat = model(pc, ma, e4, e6, rdk)
            # Weighted MSE
            loss = torch.mean(w * (yhat - y)**2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(y)
        
        val_r2, val_rmse = evaluate(model, val_loader, device)
        if val_r2 > best_r2:
            best_r2 = val_r2
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            status = "  ← best"
        else: status = ""
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss {total_loss/len(train_loader.dataset):.4f} | Val R² {val_r2:.4f}{status}")

    model.load_state_dict(torch.load(out_dir / "best_model.pt"))
    test_r2, test_rmse = evaluate(model, test_loader, device)
    print(f"\nFinal Test R²: {test_r2:.4f} | RMSE: {test_rmse:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True); parser.add_argument("--val", required=True); parser.add_argument("--test", required=True)
    parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--batch_size", type=int, default=128); parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--output_dir", default="run_qsar5_v3")
    args = parser.parse_args()
    main(args)