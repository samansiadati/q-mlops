# src/qgm/quantum_data_gen.py

import os
import json
import numpy as np
from typing import List
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer



# --- Config class for QGM parameters ---
class QGMConfig:
    def __init__(self, qubits: int, depth: int, shots: int, entangle: bool, seed: int, backend: str, noise_model=None, model_type="qgan_entangled"):
        self.qubits = qubits
        self.depth = depth
        self.shots = shots
        self.entangle = entangle
        self.seed = seed
        self.backend = backend
        self.noise_model = noise_model
        self.model_type = model_type

# --- Generate a random Hermitian matrix ---
def generate_some_matrix(n_qubits: int):
    dim = 2 ** n_qubits
    A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    H = (A + A.conj().T) / 2  # Make it Hermitian
    return H

# --- Generate a batch of quantum circuits and samples ---
def generate_batches(output_dir: str, run_id: str, epochs: int, cfg: QGMConfig):
    os.makedirs(output_dir, exist_ok=True)
    backend = Aer.get_backend(cfg.backend)

    for epoch in range(1, epochs + 1):
        # Example: create a simple variational circuit
        qc = QuantumCircuit(cfg.qubits)
        np.random.seed(cfg.seed + epoch)
        for q in range(cfg.qubits):
            qc.ry(np.random.rand() * 2 * np.pi, q)
        if cfg.entangle and cfg.qubits > 1:
            for q in range(cfg.qubits - 1):
                qc.cx(q, q + 1)
        for _ in range(cfg.depth):
            for q in range(cfg.qubits):
                qc.ry(np.random.rand() * 2 * np.pi, q)

        qc.measure_all()

        # Run simulation
        t_qc = transpile(qc, backend)
        result = backend.run(t_qc, shots=cfg.shots, noise_model=cfg.noise_model).result()
        counts = result.get_counts()

        # Convert counts to probabilities
        total_shots = sum(counts.values())
        probs = {k: v / total_shots for k, v in counts.items()}

        # Save batch to JSON
        batch_file = os.path.join(output_dir, f"{run_id}_epoch{epoch:04d}.json")
        with open(batch_file, "w") as f:
            json.dump({"epoch": epoch, "counts": counts, "probs": probs}, f)

        print(f"Generated batch for epoch {epoch} -> {batch_file}")
