import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_CSV = PROJECT_ROOT / 'results' / 'pinn3d_training_results.csv'
FIGURE_DIR = PROJECT_ROOT / 'figures'

LOSS_COLUMNS = ['Data Loss', 'PSO Loss', 'Initial Loss', 'ISO Loss', 'Total Loss']

palette = {
    'Data Loss': '#3366CC',
    'PSO Loss': '#E07A5F',
    'Initial Loss': '#3D9970',
    'ISO Loss': '#9B5DE5',
    'Total Loss': '#444444',
}


def create_figure_directory(figure_directory):
    directory = os.fspath(figure_directory) if figure_directory is not None else './figures'
    os.makedirs(directory, exist_ok=True)
    return directory


def plot_training_losses(loss_df:pd.DataFrame, save_path='pinn3d_training_loss.png', figure_directory=FIGURE_DIR):
    """
    Semilogy plot of every loss term (Data/PSO/Initial/ISO/Total) vs epoch,
    since they span several orders of magnitude over training.
    """
    figure_directory = create_figure_directory(figure_directory)
    if 'Total Loss' not in loss_df.columns:
        loss_df = loss_df.copy()
        loss_df['Total Loss'] = loss_df[['Data Loss', 'PSO Loss', 'Initial Loss', 'ISO Loss']].sum(axis=1)
    epoch = loss_df['Epoch'] if 'Epoch' in loss_df.columns else range(1, len(loss_df) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for column in LOSS_COLUMNS:
        ax.semilogy(epoch, loss_df[column], label=column, color=palette[column], linewidth=1.5)
    ax.set_title('PINN3D Training Loss Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.legend()
    ax.grid(alpha=0.3, which='both')

    out_path = os.path.join(figure_directory, save_path)
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


if __name__ == "__main__":
    loss_df = pd.read_csv(RESULTS_CSV)
    out_path = plot_training_losses(loss_df)
    print(f"Saved training loss plot to: {out_path}")
