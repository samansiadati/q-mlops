# q-mlops  
**Observability, Monitoring, and Reproducibility for Quantum Machine Learning**

q-mlops is an open-source research toolkit for **monitoring, evaluating, and reproducing Quantum Machine Learning (QML) experiments**, with a focus on **Quantum Generative Models (QGMs)** such as Variational Quantum Circuits (VQCs) and Quantum GANs (QGANs).

This project bridges **quantum computing**, **big data engineering**, and **MLOps**, enabling scalable evaluation of quantum experiments using **Apache Spark** and **MLflow**.

---

## 📄 Associated Paper

> **Monitoring and Evaluating Quantum Generative Models Using Spark and MLflow**  
> Saman Siadati  
> AAAI Fall Symposium Series (2025)

If you use this toolkit in academic work, please cite the paper.

---

## 🎯 Motivation

Quantum ML experiments suffer from:
- Limited reproducibility
- Stochastic measurement noise
- Lack of standardized evaluation pipelines
- Ad-hoc metric computation

**q-mlops** provides:
- A canonical schema for quantum experiment outputs
- Scalable metric computation
- Experiment tracking and lineage
- Reproducible evaluation across simulators and hardware

---

## 🧱 Architecture Overview
Qiskit / Quantum Backend
↓
Standardized Quantum Run Schema
↓
Spark-based Metric Computation
↓
Delta Lake (optional)
↓
MLflow Experiment Tracking


---

## 🔧 Core Features

### ✅ Standardized Quantum Experiment Schema
- Backend metadata (simulator / QPU)
- Circuit configuration
- Measurement outputs
- Noise model annotations

### 📊 Pluggable Evaluation Metrics
- Fidelity
- KL Divergence
- Jensen–Shannon Divergence
- Entanglement entropy (where applicable)
- Custom metric plugins

### 🚀 Scalable Processing
- Apache Spark DataFrames & UDFs
- Batch and (future) streaming support

### 📈 Experiment Tracking
- MLflow integration
- Parameter, metric, and artifact logging
- Git commit & environment traceability

---

## 📦 Example Usage

```python
from qmlops.metrics import fidelity
from qmlops.tracking import log_quantum_run

f = fidelity(p, q)

log_quantum_run(
    params={"circuit_depth": 4, "shots": 1024},
    metrics={"fidelity": f},
    tags={"backend": "qiskit-aer"}
)

