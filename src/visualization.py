import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd

def compute_tsne(embeddings: np.ndarray,
                 perplexity: int = 30,
                 random_state: int = 42) -> np.ndarray:
    """
    Reduce embeddings to 2D using t-SNE.
    """
    reducer = TSNE(
        n_components=2,
        learning_rate='auto',
        perplexity=perplexity,
        random_state=random_state
    )
    return reducer.fit_transform(embeddings)

def plot_tsne(coords: np.ndarray,
              labels: np.ndarray = None,
              figsize: tuple = (8, 6)):
    """
    Plot 2D t-SNE coordinates, optionally colored by labels.
    """
    plt.figure(figsize=figsize)
    if labels is None:
        plt.scatter(coords[:, 0], coords[:, 1])
    else:
        for lbl in np.unique(labels):
            mask = labels == lbl
            plt.scatter(coords[mask, 0], coords[mask, 1], label=str(lbl))
        plt.legend(title="Cluster")
    plt.title("t-SNE Visualization")
    plt.tight_layout()
    plt.show()

def visualize_hits(df_hits: pd.DataFrame, x_col: str = "score", y_col: str = "rank"):
    """
    Visualize the k-NN search hits.

    Args:
        df_hits (pd.DataFrame): DataFrame containing the search hits.
        x_col (str): Column name for the x-axis.
        y_col (str): Column name for the y-axis.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(df_hits[x_col], df_hits[y_col], c="blue", alpha=0.7)
    plt.title("k-NN Search Hits")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.show()
