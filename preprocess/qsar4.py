import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR    = "data_splits_random"
MODEL_DIR   = "models_v4/chembert_mlp"
RESULTS_DIR = "results_v4"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"
NUM_EXTRA_FEATURES = 12
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
BACKBONE_LR = 2e-5
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ── MODEL ─────────────────────────────────────────────────────────────────────
class ChemBERTaMLP(nn.Module):
    def __init__(self, chembert_model_name=MODEL_NAME, num_extra_features=NUM_EXTRA_FEATURES, hidden_dim=256):
        super(ChemBERTaMLP, self).__init__()
        self.chembert = AutoModel.from_pretrained(chembert_model_name)
        self.chembert_dim = self.chembert.config.hidden_size
        
        self.feature_projs = nn.ModuleList([nn.Linear(1, 64) for _ in range(num_extra_features)])
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.mlp = nn.Sequential(
            nn.Linear(self.chembert_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, input_ids, attention_mask, extra_features):
        outputs = self.chembert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Encode extra features: apply each feature's own linear projection to preserve identity
        x_list = [proj(extra_features[:, i].unsqueeze(-1)) for i, proj in enumerate(self.feature_projs)]
        x = torch.stack(x_list, dim=1) # (batch_size, num_extra_features, 64)
        
        x = self.transformer_encoder(x)
        encoded_features = x.mean(dim=1)
        
        # Concatenate
        combined = torch.cat((cls_embedding, encoded_features), dim=1)
        
        # Predict
        out = self.mlp(combined)
        return out.squeeze(-1)

# ── DATASET ───────────────────────────────────────────────────────────────────
class QsarDataset(Dataset):
    def __init__(self, df, feature_cols, tokenizer, max_len=MAX_LEN):
        self.smiles = df["smiles"].values
        self.features = df[feature_cols].values.astype(np.float32)
        self.targets = df["pic50"].values.astype(np.float32) if "pic50" in df.columns else None
        
        # Add weights based on zone to penalize tail errors
        self.weights = None
        if "zone" in df.columns:
            zone_weights = {"low": 4.0, "bulk": 1.0, "high": 3.0}
            self.weights = np.array([zone_weights.get(z, 1.0) for z in df["zone"].values], dtype=np.float32)
            
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.smiles)
        
    def __getitem__(self, idx):
        smile = str(self.smiles[idx])
        encoding = self.tokenizer(
            smile,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'features': torch.tensor(self.features[idx], dtype=torch.float32)
        }
        
        if self.targets is not None:
            item['target'] = torch.tensor(self.targets[idx], dtype=torch.float32)
            if self.weights is not None:
                item['weight'] = torch.tensor(self.weights[idx], dtype=torch.float32)
            
        return item

# ── TRAINING FUNCTION ─────────────────────────────────────────────────────────
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        features = batch['features'].to(device)
        targets = batch['target'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, features)
        
        if 'weight' in batch:
            weights = batch['weight'].to(device)
            loss = torch.mean(weights * (outputs - targets)**2)
        else:
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(input_ids, attention_mask, features)
            
            if 'weight' in batch:
                weights = batch['weight'].to(device)
                loss = torch.mean(weights * (outputs - targets)**2)
            else:
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    r2 = r2_score(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    return total_loss / len(dataloader), r2, rmse, np.array(all_preds)

# ── MAIN RUNNER ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Loading data...")
    
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    # Filter synthetics
    train_df = train_df[train_df["smiles"] != "__synthetic__"].copy()
    
    # Extract 12 PhysChem features
    with open(os.path.join(DATA_DIR, "feature_cols.txt")) as f:
        FEATURE_COLS = [l.strip() for l in f if l.strip()]
    PHYSCHEM_COLS = [c for c in FEATURE_COLS if not c.startswith("fp_") and not c.startswith("fcfp_")]
    
    # Use whatever features are available instead of hardcoding 12
    NUM_EXTRA_FEATURES = len(PHYSCHEM_COLS)
    selected_features = PHYSCHEM_COLS
    print(f"Selected {NUM_EXTRA_FEATURES} extra features: {selected_features}")
    
    # Fill any NaNs
    train_df[selected_features] = train_df[selected_features].fillna(0)
    val_df[selected_features] = val_df[selected_features].fillna(0)
    test_df[selected_features] = test_df[selected_features].fillna(0)
    
    # Initialize tokenizer and model
    print("Initializing tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = ChemBERTaMLP(chembert_model_name=MODEL_NAME, num_extra_features=NUM_EXTRA_FEATURES).to(DEVICE)
    
    # Create datasets and dataloaders
    train_dataset = QsarDataset(train_df, selected_features, tokenizer)
    val_dataset = QsarDataset(val_df, selected_features, tokenizer)
    test_dataset = QsarDataset(test_df, selected_features, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    criterion = nn.MSELoss()
    optimizer_grouped_parameters = [
        {'params': model.chembert.parameters(), 'lr': BACKBONE_LR},
        {'params': model.feature_projs.parameters(), 'lr': LEARNING_RATE},
        {'params': model.transformer_encoder.parameters(), 'lr': LEARNING_RATE},
        {'params': model.mlp.parameters(), 'lr': LEARNING_RATE}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    
    print("=" * 60)
    print("Starting Training...")
    
    best_val_rmse = float('inf')
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_r2, val_rmse, _ = eval_model(model, val_loader, criterion, DEVICE)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f}")
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))
            print("  --> Saved new best model")
            
    print("=" * 60)
    print("Evaluating on Test Set with Best Model...")
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pth"), weights_only=True))
    
    test_loss, test_r2, test_rmse, test_preds = eval_model(model, test_loader, criterion, DEVICE)
    test_mae = mean_absolute_error(test_df["pic50"].values, test_preds)
    
    print(f"Final Test R²   : {test_r2:.4f}")
    print(f"Final Test RMSE : {test_rmse:.4f}")
    print(f"Final Test MAE  : {test_mae:.4f}")
    
    # Save Report
    report_path = os.path.join(RESULTS_DIR, "qsar4_report_chemberta.txt")
    with open(report_path, "w") as f:
        f.write("QSAR 4: ChemBERTa + 12 Features + MLP Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test R²   : {test_r2:.4f}\n")
        f.write(f"Test RMSE : {test_rmse:.4f}\n")
        f.write(f"Test MAE  : {test_mae:.4f}\n")
        
    print(f"\nReport saved to: {report_path}")