# 🦿 Intelligent Gait Recognition for Smart Prosthetics using Deep Learning

## 📘 Overview
This project presents a unified deep learning framework for **human gait recognition** using the **HuGaDB v2 (Human Gait Database)** wearable IMU dataset.  
It aims to enable **adaptive prosthetic control** by accurately classifying lower-limb activities through sensor-based motion data.

A series of experiments (EXP-001 through EXP-009) were conducted to compare major time-series architectures including **LSTM**, **BiLSTM**, **GRU**, **CNN-LSTM**, **TCN**, **Transformer**, and **ResNet1D**, using a consistent preprocessing and training pipeline.

---

## 🧩 System Architecture

```text
 ┌─────────────────────────┐
 │ Wearable IMU Sensors    │
 │ (Foot, Shank, Thigh)    │
 └───────────┬─────────────┘
             │
             ▼
 ┌─────────────────────────┐
 │ Preprocessing &         │
 │ Segmentation            │
 │ (2s windows, 50% overlap,
 │  z-score normalization) │
 └───────────┬─────────────┘
             │
             ▼
 ┌─────────────────────────┐
 │ Deep Learning Models    │
 │ (LSTM / GRU / TCN / etc.)│
 │ Activity Classification │
 └───────────┬─────────────┘
             │
             ▼
 ┌─────────────────────────┐
 │ Prosthetic Control      │
 │ (Adaptive Actuation)    │
 └─────────────────────────┘
```

---

## 🧠 Motivation
Current prosthetic systems rely on fixed rule-based control, leading to latency and poor adaptation across users.  
This project explores **sensor-driven deep learning** for real-time gait recognition, providing the basis for an intelligent control module capable of adjusting to user intent and walking dynamics.

---

## 📂 Repository Structure
```
project_root/
│
├── configs/
│   ├── config_exp001.yaml
│   ├── config_exp002.yaml
│   └── ... config_exp009.yaml
│
├── data/
│   └── HuGaDB/              # raw and preprocessed sensor data
│
├── models/
│   ├── lstm.py
│   ├── bilstm.py
│   ├── gru.py
│   ├── cnn_lstm.py
│   ├── tcn.py
│   ├── transformer.py
│   └── resnet1d.py
│
├── utils/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── metrics.py
│   └── model_factory.py
│
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
└── references.bib
```

---

## ⚙️ Setup Instructions

### 1️⃣ Environment Setup
```bash
conda create -n gait python=3.10
conda activate gait
pip install -r requirements.txt
```

### 2️⃣ Dataset
Download **HuGaDB v2** dataset from Kaggle:  
👉 [https://www.kaggle.com/datasets/romanchereshnev/hugadb-human-gait-database](https://www.kaggle.com/datasets/romanchereshnev/hugadb-human-gait-database)

Place the extracted folder under:
```
data/HuGaDB/
```

### 3️⃣ Running an Experiment
```bash
python train.py --config configs/config_exp003.yaml
```

Trained models, logs, and plots will be stored automatically in:
```
artifacts/exp003/
```

---

## 🧪 Experiments Summary

| Experiment | Model Type  | Key Features | Accuracy | Macro F1 | Notes |
|-------------|-------------|--------------|-----------|-----------|-------|
| EXP-001 | LSTM | 2 layers, 128 hidden | 0.861 | 0.792 | Baseline recurrent model |
| EXP-002 | BiLSTM | Bidirectional variant | 0.866 | 0.801 | Slight improvement over LSTM |
| EXP-005 | GRU | Gated recurrent unit | **0.874** | **0.809** | Best overall performer |
| EXP-007 | TCN | Temporal convolutional | 0.854 | 0.788 | Efficient but less stable |
| EXP-009 | Transformer | Self-attention model | 0.847 | 0.781 | High variance, overfit risk |

---

## 📊 Evaluation Metrics
- **Accuracy**
- **Precision / Recall / F1 (macro & weighted)**
- **Confusion Matrix**
- **Training vs Validation Loss Curves**
- **Parameter Count & Inference Time**

---

## 🔬 Key Findings
- Recurrent architectures (GRU/LSTM) perform comparably or better than attention-based ones under resource constraints.  
- GRU achieves high accuracy with fewer parameters → ideal for **embedded prosthetic hardware**.  
- Temporal convolutional models (TCN) offer robust parallelism but require fine-tuning of dilation factors.  
- Transformers benefit from longer input windows but tend to overfit smaller datasets.

---

## 🚀 Future Work
- Integrate the best-performing model (GRU) into a **closed-loop prosthetic controller**.  
- Perform **real-time inference testing** on embedded boards (e.g., NVIDIA Jetson Nano).  
- Explore **transfer learning** across subjects and sensor setups.  
- Extend to **multimodal fusion** with EMG and pressure sensors.

---

## 🧾 Citation
If you use this repository or its experimental framework, please cite:

```
@inproceedings{kataria2025gait,
  title={Intelligent Gait Recognition for Smart Prosthetics using Deep Learning},
  author={Kataria, Sujaan and Basu, Shatabdi},
  booktitle={Proceedings of the IEEE International Conference on Biomedical Systems},
  year={2025}
}
```

---

## 🧑‍💻 Authors
**Sujaan Kataria** – B.Tech CSE (Data Science), Manipal University Jaipur  
**Dr. Shatabdi Basu** – Associate Professor, Department of CSE, MUJ  

---

## 📚 References
See full list in [`references.bib`](references.bib).

---

## 🧩 License
This project is released under the **MIT License**.  
You are free to use, modify, and distribute it for academic or research purposes with appropriate attribution.

---

> **Last Updated:** October 2025  
> **Repository Maintainer:** Sujaan Kataria (sujaan.kataria@learner.muj.edu.in)
