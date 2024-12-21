import numpy as np
import matplotlib.pyplot as plt

def plot_features(features, labels):
    """
    Plot extracted features with labels.
    
    Args:
        features (numpy.ndarray): Feature matrix
        labels (numpy.ndarray): Corresponding labels
    """
    plt.figure(figsize=(12, 6))
    for i in range(features.shape[1]):
        plt.subplot(2, 3, i+1)
        plt.scatter(labels, features[:, i])
        plt.xlabel('Tempo/Rhythm Label')
        plt.ylabel(f'Feature {i+1}')
    plt.tight_layout()
    plt.show()

def save_predictions(predictions, filepath):
    """
    Save predictions to a file.
    
    Args:
        predictions (list): List of predictions
        filepath (str): Output file path
    """
    np.save(filepath, np.array(predictions))