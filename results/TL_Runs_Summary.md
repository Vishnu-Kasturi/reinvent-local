# PDL1_PH_D Transfer Learning Experiments Summary

This document summarizes the 6 Transfer Learning (TL) runs conducted to optimize the `freeze_n_layers` and `lr` (learning rate) parameters for the `pdl1_ph_d` dataset. The goal was to find a "Goldilocks" configuration that learns the target chemistry without completely collapsing into a memorized state (which manifests as a highly right-skewed Tanimoto similarity distribution vs the training set).

## Summary Table

| Run | `freeze_n_layers` | `lr` | Epochs | End Mean Tanimoto | Verdict |
|-----|-------------------|------|--------|-------------------|---------|
| **1** | 1 (Embedding only) | 5e-5 | 150 | 0.61 | ❌ Memorized quickly |
| **2** | 3 (Emb + LSTM0+1) | 3e-5 | 150 | 0.28 | ❌ Learned nothing |
| **3** | **2 (Emb + LSTM0)** | **5e-5** | 150 | **0.46** | **✅ Best configuration (Goldilocks)** |
| **4** | 2 | 1e-4 | 150 | 0.75 | ❌ Memorized extremely fast |
| **5** | 2 | 1e-5 | 400 | 0.29 | ❌ Learned too slowly |
| **6** | 2 | 5e-5 | 250 | 0.62 | ❌ Pushed optimal config too far; eventual memorization |

> **Recommendation**: The optimal checkpoint to use for downstream Mol2Mol or RL is **Run 3 / Run 6 at Epoch 80-100**. At this stage, the model generates a bimodal distribution: it can produce both completely novel scaffolds and highly target-similar structures.

---

## Detailed Run Reports

### Run 1: Memorization
- **Config**: `freeze_n_layers=1`, `lr=5e-5`, `num_epochs=150`
- **Result**: Tanimoto similarity to the training set shoots up rapidly after epoch 60. By epoch 150, it is heavily right-skewed, indicating the model has lost its prior diversity and is just spitting out training set variants.
![Run 1 Tanimoto](file:///Users/vishnukasturi/Intern/reinvent-local/results/pdl1_ph_d_tl_epoch_tanimoto.png)
![Run 1 KDE](file:///Users/vishnukasturi/Intern/reinvent-local/results/pdl1_ph_d_tl_epoch_kde_vs_baseline.png)

### Run 2: Over-frozen
- **Config**: `freeze_n_layers=3`, `lr=3e-5`, `num_epochs=150`
- **Result**: The model is too rigid. The Tanimoto distribution stays pinned to the left (~0.28), identical to the untrained prior. It fails to learn the target chemistry.
![Run 2 Tanimoto](file:///Users/vishnukasturi/Intern/reinvent-local/results/pdl1_ph_d_tl_run2_epoch_tanimoto.png)
![Run 2 KDE](file:///Users/vishnukasturi/Intern/reinvent-local/results/pdl1_ph_d_tl_run2_epoch_kde_vs_baseline.png)

### Run 3: The Goldilocks Setup
- **Config**: `freeze_n_layers=2`, `lr=5e-5`, `num_epochs=150`
- **Result**: Perfect balance. Between epochs 80 and 100, the mean Tanimoto pulls away from the median, creating a bimodal curve.
![Run 3 Tanimoto](file:///Users/vishnukasturi/Intern/reinvent-local/results/pdl1_ph_d_tl_run3_epoch_tanimoto.png)
![Run 3 KDE](file:///Users/vishnukasturi/Intern/reinvent-local/results/pdl1_ph_d_tl_run3_epoch_kde_vs_baseline.png)

### Run 4: High Learning Rate
- **Config**: `freeze_n_layers=2`, `lr=1e-4`, `num_epochs=150`
- **Result**: Increasing the LR causes catastrophic overfitting. The median Tanimoto hits 1.000 by epoch 140, meaning the model is exclusively generating exact copies of the training set.
![Run 4 Tanimoto](file:///Users/vishnukasturi/Intern/reinvent-local/results/pdl1_ph_d_tl_run4_epoch_tanimoto.png)
![Run 4 KDE](file:///Users/vishnukasturi/Intern/reinvent-local/results/pdl1_ph_d_tl_run4_epoch_kde_vs_baseline.png)

### Run 5: Low Learning Rate (Extended)
- **Config**: `freeze_n_layers=2`, `lr=1e-5`, `num_epochs=400`
- **Result**: Decreasing the LR makes the model learn too slowly. Even when given 400 epochs, the Tanimoto barely creeps up to 0.29. It effectively mirrors Run 2 (learned nothing).
![Run 5 Tanimoto](file:///Users/vishnukasturi/Intern/reinvent-local/results/pdl1_ph_d_tl_run5_epoch_tanimoto.png)
![Run 5 KDE](file:///Users/vishnukasturi/Intern/reinvent-local/results/pdl1_ph_d_tl_run5_epoch_kde_vs_baseline.png)

### Run 6: Optimal Config (Extended)
- **Config**: `freeze_n_layers=2`, `lr=5e-5`, `num_epochs=250`
- **Result**: Confirms that the bimodal distribution in Run 3 (Epoch 80-120) is a transient state. If training is allowed to proceed to 250 epochs, the model inevitably collapses into memorization.
![Run 6 Tanimoto](file:///Users/vishnukasturi/Intern/reinvent-local/results/pdl1_ph_d_tl_run6_epoch_tanimoto.png)
![Run 6 KDE](file:///Users/vishnukasturi/Intern/reinvent-local/results/pdl1_ph_d_tl_run6_epoch_kde_vs_baseline.png)
