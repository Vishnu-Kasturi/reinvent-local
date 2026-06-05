import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    repo = "/Users/vishnukasturi/Intern/reinvent-local"
    csv_path = f"{repo}/Navneet_Git/mol2mol_results/medium_similarity/top15_pairs.csv"
    out_png = f"{repo}/Navneet_Git/mol2mol_results/medium_similarity/pair_shifts_1to1.png"
    brain_png = "/Users/vishnukasturi/.gemini/antigravity/brain/b1740afb-b51b-4bea-901b-35388d53206f/artifacts/medium_sim_1to1_shifts.png"
    
    print("[*] Loading pairs data...")
    df = pd.read_csv(csv_path)
    
    # We want to order them as in the image (or by mol_id)
    # The image has mol330 at the bottom and mol282 at the top. Let's sort by index ascending so that it plots top to bottom correctly.
    # Actually, matplotlib plots y=0 at bottom. So to have rank 1 at the top, we need to reverse the order for the y-axis.
    df = df.iloc[::-1].reset_index(drop=True)
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PD1-PDL1 Leads (Top 15) — Original vs. Optimized Pairs (Medium Similarity)', fontsize=16, fontweight='bold', y=0.98)
    
    color_orig = '#2ecc71' # Green
    color_opt = '#e74c3c'  # Red
    color_target = '#f1c40f' # Yellow
    
    # -----------------------------------------
    # Top-Left: KDE pIC50
    # -----------------------------------------
    sns.kdeplot(data=df, x='lead_pic50', color=color_orig, fill=True, alpha=0.2, linewidth=2, ax=axes[0,0])
    sns.kdeplot(data=df, x='optimized_pic50', color=color_opt, fill=True, alpha=0.2, linewidth=2, ax=axes[0,0])
    
    # Custom legend for KDE
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color=color_orig, lw=3),
        Line2D([0], [0], color=color_opt, lw=3),
        Line2D([0], [0], color=color_target, lw=2, linestyle='--')
    ]
    axes[0,0].legend(custom_lines, ['Original Leads', 'Optimized Analogues', 'Target pIC50: 8.5'], loc='upper left')
    
    axes[0,0].axvline(8.5, color=color_target, linestyle='--', zorder=0)
    axes[0,0].set_title('pIC50 Overall Distribution', fontweight='bold')
    axes[0,0].set_xlabel('pIC50')
    axes[0,0].set_ylabel('Density')
    
    # -----------------------------------------
    # Top-Right: KDE Solubility
    # -----------------------------------------
    sns.kdeplot(data=df, x='lead_solubility', color=color_orig, fill=True, alpha=0.2, linewidth=2, ax=axes[0,1])
    sns.kdeplot(data=df, x='optimized_solubility', color=color_opt, fill=True, alpha=0.2, linewidth=2, ax=axes[0,1])
    
    custom_lines_sol = [
        Line2D([0], [0], color=color_orig, lw=3),
        Line2D([0], [0], color=color_opt, lw=3),
        Line2D([0], [0], color=color_target, lw=2, linestyle='--')
    ]
    axes[0,1].legend(custom_lines_sol, ['Original Leads', 'Optimized Analogues', 'TOML Threshold: -3.0'], loc='upper right')
    
    axes[0,1].axvline(-3.0, color=color_target, linestyle='--', zorder=0)
    axes[0,1].set_title('Solubility (logS) Overall Distribution', fontweight='bold')
    axes[0,1].set_xlabel('logS')
    axes[0,1].set_ylabel('Density')
    
    # -----------------------------------------
    # Bottom-Left: 1-to-1 pIC50 Shifts
    # -----------------------------------------
    y_positions = np.arange(len(df))
    axes[1,0].hlines(y=y_positions, xmin=df[['lead_pic50', 'optimized_pic50']].min(axis=1), xmax=df[['lead_pic50', 'optimized_pic50']].max(axis=1), color='grey', alpha=0.5)
    axes[1,0].scatter(df['lead_pic50'], y_positions, color=color_orig, s=80, label='Original', zorder=3)
    axes[1,0].scatter(df['optimized_pic50'], y_positions, color=color_opt, s=80, label='Optimized', zorder=3)
    axes[1,0].axvline(8.5, color=color_target, linestyle='--', zorder=0)
    axes[1,0].set_yticks(y_positions)
    axes[1,0].set_yticklabels(df['mol_id'])
    axes[1,0].set_title('1-to-1 pIC50 Shifts', fontweight='bold')
    axes[1,0].set_xlabel('pIC50')
    axes[1,0].legend(loc='lower left')
    
    # -----------------------------------------
    # Bottom-Right: 1-to-1 Solubility Shifts
    # -----------------------------------------
    axes[1,1].hlines(y=y_positions, xmin=df[['lead_solubility', 'optimized_solubility']].min(axis=1), xmax=df[['lead_solubility', 'optimized_solubility']].max(axis=1), color='grey', alpha=0.5)
    axes[1,1].scatter(df['lead_solubility'], y_positions, color=color_orig, s=80, label='Original', zorder=3)
    axes[1,1].scatter(df['optimized_solubility'], y_positions, color=color_opt, s=80, label='Optimized', zorder=3)
    axes[1,1].axvline(-3.0, color=color_target, linestyle='--', zorder=0)
    axes[1,1].set_yticks(y_positions)
    axes[1,1].set_yticklabels(df['mol_id'])
    axes[1,1].set_title('1-to-1 Solubility (logS) Shifts', fontweight='bold')
    axes[1,1].set_xlabel('logS')
    axes[1,1].legend(loc='lower right')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    plt.savefig(out_png, dpi=300)
    plt.savefig(brain_png, dpi=300)
    print(f"[+] Saved 1-to-1 shift plot to:\n    - {out_png}\n    - {brain_png}")

if __name__ == "__main__":
    main()
