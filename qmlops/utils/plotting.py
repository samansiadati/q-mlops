# src/utils/plotting.py
import matplotlib.pyplot as plt

def plot_probs(probs, ref, title, path):
    """
    Simple plotting function for probability distributions.
    probs: list of floats (predicted distribution)
    ref: list of floats (reference distribution)
    """
    plt.figure()
    plt.plot(probs, label="Generated")
    plt.plot(ref, label="Reference")
    plt.title(title)
    plt.legend()
    plt.savefig(path)
    plt.close()
