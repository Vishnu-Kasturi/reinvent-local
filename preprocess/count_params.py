import torch
from qsar4 import ChemBERTaMLP, MODEL_NAME

model = ChemBERTaMLP(chembert_model_name=MODEL_NAME, num_extra_features=18)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")

# Breakdown
print(f"ChemBERTa backbone: {sum(p.numel() for p in model.chembert.parameters()):,}")
print(f"Feature Projections: {sum(p.numel() for p in model.feature_projs.parameters()):,}")
print(f"Transformer Encoder: {sum(p.numel() for p in model.transformer_encoder.parameters()):,}")
print(f"MLP Head: {sum(p.numel() for p in model.mlp.parameters()):,}")
