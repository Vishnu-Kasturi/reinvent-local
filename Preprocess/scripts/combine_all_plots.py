import os, sys
import matplotlib.pyplot as plt

REPO_ROOT = "/Users/vishnukasturi/Intern/reinvent-local"
PLOTS_DIR = os.path.join(REPO_ROOT, "Navneet_Git", "TL_plots")

def main():
    img_tanimoto = os.path.join(PLOTS_DIR, "jak2_tl_epoch_tanimoto.png")
    img_tl_kde = os.path.join(PLOTS_DIR, "jak2_tl_epoch_kde_vs_baseline.png")
    img_rl_kde = os.path.join(PLOTS_DIR, "jak2_rl_kde_vs_baseline.png")
    
    if not (os.path.exists(img_tanimoto) and os.path.exists(img_tl_kde) and os.path.exists(img_rl_kde)):
        print("Error: One or more source PNG files are missing.")
        sys.exit(1)
        
    print("Loading plots...")
    im1 = plt.imread(img_tanimoto)
    im2 = plt.imread(img_tl_kde)
    im3 = plt.imread(img_rl_kde)
    
    # Create composite figure: 2 rows, 2 columns
    # Row 1: Tanimoto (left), TL KDE (right)
    # Row 2: RL KDE (spanning both columns or centered)
    fig = plt.figure(figsize=(24, 20), dpi=120)
    
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax1.imshow(im1)
    ax1.axis('off')
    ax1.set_title("A. Transfer Learning Checkpoints: Max Tanimoto Similarity", fontsize=18, weight='bold', pad=10)
    
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax2.imshow(im2)
    ax2.axis('off')
    ax2.set_title("B. Transfer Learning Checkpoints: Property KDE Evolution vs Baseline", fontsize=18, weight='bold', pad=10)
    
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    ax3.imshow(im3)
    ax3.axis('off')
    ax3.set_title("C. Reinforcement Learning: MPO Candidates vs Baseline & TL", fontsize=18, weight='bold', pad=10)
    
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "jak2_pipeline_all_plots.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=120)
    plt.close()
    print(f"[+] Saved composite pipeline plot to {out_path}")

if __name__ == "__main__":
    main()
