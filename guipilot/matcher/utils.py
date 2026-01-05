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
    """
    Generates a heat-map visualization of the widget matching score matrix.

    This utility function renders the 2D grid of similarity scores between two
    sets of widgets and overlays the optimal matching path (e.g., calculated
    via Longest Matching Subsequence) to provide visual insight into the
    alignment logic.

    Args:
        scores (np.ndarray): A 2D NumPy array of shape (M, N) containing the
            similarity scores between widgets from the original (M) and
            mutated (N) screens.
        path (list[tuple]): A list of index pairs `(i, j)` representing the
            optimal alignment path through the scoring matrix.
        widget_keys_i (list[int]): A list of unique widget IDs from the
            original/source screen, used for y-axis labeling.
        widget_keys_j (list[int]): A list of unique widget IDs from the
            mutated/target screen, used for x-axis labeling.

    Returns:
        tuple[Figure, Axes]: A tuple containing the Matplotlib Figure and Axes
            objects for further customization or saving.
    """

    fig, ax = plt.subplots(figsize=(12, 10))

    # Display the score grid as a heatmap using the viridis colormap
    ax.imshow(scores, cmap="viridis", interpolation="nearest")

    # Highlight the optimal matching path with a red line and circular markers
    x_path, y_path = zip(*path)
    ax.plot(y_path, x_path, color="red", linewidth=2, marker="o")

    # Annotate each cell in the grid with its rounded score value
    for (i, j), val in np.ndenumerate(scores):
        val = round(val, 2)
        ax.text(j, i, f"{val}", ha="center", va="center", color="white", fontsize=8)

    # Set x and y ticks and labels using the provided widget IDs
    ax.set_xticks(np.arange(scores.shape[1]), [x for x in widget_keys_j])
    ax.set_xticklabels(widget_keys_j)
    ax.set_yticks(np.arange(scores.shape[0]), [y for y in widget_keys_i])
    ax.set_yticklabels(widget_keys_i)

    # Add visual context including a color bar and axis labels
    fig.colorbar(ax.images[0], label="Similarity Score")
    ax.set_ylabel("Original Screen Widget IDs")
    ax.set_xlabel("Mutated Screen Widget IDs")
    ax.set_title("2D Grid of Scores with Highlighted Matching Path")

    return fig, ax
