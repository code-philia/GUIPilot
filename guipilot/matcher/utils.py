import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def visualize_match_scores(
    scores: np.ndarray,
    path: list[tuple],
    widget_keys_i: list[int],
    widget_keys_j: list[int],
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(12, 10))

    # Display the grid
    ax.imshow(scores, cmap="viridis", interpolation="nearest")

    # Highlight the path
    x_path, y_path = zip(*path)
    ax.plot(y_path, x_path, color="red", linewidth=2, marker="o")

    # Annotate the grid with values
    for (i, j), val in np.ndenumerate(scores):
        val = round(val, 2)
        ax.text(j, i, f"{val}", ha="center", va="center", color="white", fontsize=8)

    # Set x and y ticks using widget keys
    ax.set_xticks(np.arange(scores.shape[1]), [x for x in widget_keys_j])
    ax.set_xticklabels(widget_keys_j)
    ax.set_yticks(np.arange(scores.shape[0]), [y for y in widget_keys_i])
    ax.set_yticklabels(widget_keys_i)

    # Add color bar, labels, and title
    fig.colorbar(label="Score")
    ax.set_ylabel("original")
    ax.set_xlabel("mutated")
    ax.set_title("2D Grid of Scores with Highlighted Path")

    return fig, ax
