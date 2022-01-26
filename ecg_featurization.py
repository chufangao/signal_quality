import numpy as np

def featurize_beats(beats):
    ### beats is list of 1D numpy arrays (can be of varying length)
    means = [np.mean(b) for b in beats]
    stds = [np.std(b) for b in beats]

    return np.stack([means, stds], axis=-1)

