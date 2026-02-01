# src/qgm/noise_models.py

from qiskit_aer.noise import NoiseModel

def simple_noise_model():
    """
    Returns a placeholder noise model.
    Right now, this is just an empty NoiseModel (ideal simulation).
    You can later add real noise channels if needed.
    """
    return NoiseModel()
