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

