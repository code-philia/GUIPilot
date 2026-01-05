import random

import numpy as np


def sample_p(population: dict | list, p: float) -> dict | list:
    """
    Args:
        population: The list of elements to sample from.
        p: The percentage of elements to select (0, 1].

    Returns:
        A list containing the randomly selected elements.
    """
    k = max(int(p * len(population)), 1)
    if len(population) < k:
        return population
    if isinstance(population, dict):
        keys = random.sample(list(population), k)
        values = [population[k] for k in keys]
        return dict(zip(keys, values))
    elif isinstance(population, list):
        return random.sample(population, k)


def get_context_color(widget_image: np.ndarray) -> tuple[int, int, int]:
    """Find the most common color in the image as an RGB tuple"""
    col_range = (256, 256, 256)
    a2D = widget_image.reshape(-1, widget_image.shape[-1])
    a1D = np.ravel_multi_index(a2D.T, col_range)
    try:
        return np.unravel_index(np.bincount(a1D).argmax(), col_range)
    except ValueError:
        return (255, 255, 255)
