# 🚦 Reality → Simulation

## AI-Powered Real-World Traffic Digital Twin

**Reality → Perception → Understanding → Prediction → Simulation → Optimization**

Reality → Simulation is an end-to-end AI and traffic simulation project that transforms real-world traffic video into structured vehicle trajectories, learns vehicle motion patterns, evaluates interaction-aware prediction models, reconstructs a microscopic traffic simulation in SUMO, and evaluates counterfactual traffic-signal interventions.

The project combines **Computer Vision, Multi-Object Tracking, Time-Series Deep Learning, Graph Neural Networks, Microscopic Traffic Simulation, and Optimization** into a single reproducible pipeline.

---

## 🎯 Project Objective

Traditional traffic analysis is often limited to observing what has already happened.

This project explores a different question:

> **Can real-world traffic observations be transformed into a digital twin that can predict vehicle behavior and evaluate what might happen under alternative traffic-control strategies?**

The system therefore follows the pipeline:

```text
Real-World Traffic Video
          ↓
Vehicle Detection
          ↓
Multi-Object Tracking
          ↓
Vehicle Trajectories
          ↓
Motion Feature Extraction
          ↓
Trajectory Prediction
          ↓
Interaction Modeling
          ↓
SUMO Digital Twin
          ↓
Counterfactual Simulation
          ↓
Signal Optimization