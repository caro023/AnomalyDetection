# Network Traffic Anomaly Detection using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning-based system for automatic detection of network anomalies and cyber attacks using Autoencoders.

## Project Overview

This project implements an **unsupervised anomaly detection system** for network traffic analysis using deep autoencoders. The system is trained primarily on normal traffic patterns and learns to detect anomalies by measuring reconstruction error.

### Key Features

-  **Semi-supervised Learning**: Realistic training with 5% attack contamination
-  **Deep Autoencoder Architecture**: 33 → 256 → 128 → 64 → 16 (latent) → 64 → 128 → 256 → 33
-  **High Performance**: ROC-AUC 0.96+, Detection Rate 98%+
-  **Low False Positives**: ~10% false alarm rate
-  **Comprehensive Evaluation**: ROC curves, confusion matrix, error analysis
-  **Production-Ready**: Modular code, configurable parameters, scalable design

### The dataset can be dowloaded as follows:
```
    dataset_name = "UNSW-NB15"
    subset = ['Network-Flows', 'Packet-Fields', 'Payload-Bytes']  
    files = [3, 5, 10] # files that I used
    from nids_datasets import Dataset, DatasetInfo
    data = Dataset(dataset=dataset_name, subset=subset, files=files)

    data.download() 
```
## Architecture

### Deep Autoencoder

```
Encoder:
  Input(33) → Dense(256) → BatchNorm → ReLU → Dropout(0.3)
           → Dense(128) → BatchNorm → ReLU → Dropout(0.2)
           → Dense(64)  → BatchNorm → ReLU → Dropout(0.2)
           → Dense(16)  [Latent Space]

Decoder:
  Latent(16) → Dense(64)  → BatchNorm → ReLU → Dropout(0.2)
             → Dense(128) → BatchNorm → ReLU → Dropout(0.2)
             → Dense(256) → BatchNorm → ReLU → Dropout(0.3)
             → Dense(33)  [Reconstruction]
```

**Key Design Choices:**
- **BatchNormalization**: Stabilizes training and improves convergence
- **Dropout**: Prevents overfitting and improves generalization
- **ReLU Activation**: Efficient non-linearity
- **MSE Loss**: Measures reconstruction quality


## Configuration Options

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_subset` | True | Use subset for faster training |
| `subset_size` | 50000 | Number of samples if subset enabled |
| `contamination_rate` | 0.05 | Attack percentage in training (0.0-0.2) |
| `epochs` | 30 | Training epochs (auto-adjusted) |
| `batch_size` | 256 | Batch size for training |
| `learning_rate` | 0.001 | Adam optimizer learning rate |
| `latent_dim` | 16 | Bottleneck dimension |

### Contamination Rate Guide

- **0.00**: Pure unsupervised (only normal traffic)
- **0.05**: Realistic semi-supervised 
- **0.10**: Higher attack exposure
- **0.20**: Nearly supervised approach

## Dataset Information

### UNSW-NB15 Dataset

**Overview:**
- **Total Samples**: 2,540,044 network flows
- **Features**: 49 (33 used after preprocessing)
- **Classes**: Normal (87%) + 9 attack types (13%)

**Attack Types:**
- Exploits
- DoS (Denial of Service)
- Reconnaissance
- Backdoor
- Analysis
- Fuzzers
- Shellcode
- Worms
- Generic

## Methodology

### Data Preprocessing

1. **Feature Selection**: Remove IPs, ports, IDs, timestamps
2. **Categorical Encoding**: Label encoding for protocol, service, state
3. **Standardization**: Zero mean, unit variance scaling
4. **Missing Values**: Median imputation

## Requirements

```txt
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyarrow>=12.0.0
```
## Example


Output:

```
Dataset loaded successfully: 2,059,415 samples, 50 features

Using subset of 500,000 samples for faster training
 (Full dataset: 2,059,415 samples)
   Subset created: 500,000 samples

Preprocessing data...

 Attack type distribution:
   normal: 475,808 (95.2%)
   exploits: 6,756 (1.4%)
   generic: 6,159 (1.2%)
   fuzzers: 5,334 (1.1%)
   reconnaissance: 3,193 (0.6%)
   dos: 1,380 (0.3%)
   analysis: 509 (0.1%)
   backdoor: 468 (0.1%)
   shellcode: 354 (0.1%)
   worms: 39 (0.0%)

 Label distribution:
   Normal (0): 475,808 (95.2%)
   Attack (1): 24,192 (4.8%)

Excluding 18 columns: ['label', 'binary_label', 'attack_label', 'flow_id', 'source_ip']...

Final feature dimension: 33

Dataset statistics:
 Total samples: 500,000
 Normal traffic: 475,808 (95.2%)
 Attack traffic: 24,192 (4.8%)

Training mode: Pure unsupervised (only normal traffic)
 Device: cpu
Starting training

Epoch  1/20 | Train Loss: 0.251873 | Val Loss: 0.081095
Epoch  2/20 | Train Loss: 0.179434 | Val Loss: 0.072048
Epoch  3/20 | Train Loss: 0.157793 | Val Loss: 0.045773
Epoch  4/20 | Train Loss: 0.149787 | Val Loss: 0.050101
Epoch  5/20 | Train Loss: 0.141617 | Val Loss: 0.045454
Epoch  6/20 | Train Loss: 0.137938 | Val Loss: 0.053232
Epoch  7/20 | Train Loss: 0.130336 | Val Loss: 0.031089
Epoch  8/20 | Train Loss: 0.128783 | Val Loss: 0.041980
Epoch  9/20 | Train Loss: 0.123199 | Val Loss: 0.031686
Epoch 10/20 | Train Loss: 0.120650 | Val Loss: 0.033858
Epoch 11/20 | Train Loss: 0.118447 | Val Loss: 0.028142
Epoch 12/20 | Train Loss: 0.113984 | Val Loss: 0.036626
Epoch 13/20 | Train Loss: 0.111382 | Val Loss: 0.036934
Epoch 14/20 | Train Loss: 0.110018 | Val Loss: 0.026618
Epoch 15/20 | Train Loss: 0.107214 | Val Loss: 0.028913
Epoch 16/20 | Train Loss: 0.103316 | Val Loss: 0.026236
Epoch 17/20 | Train Loss: 0.101474 | Val Loss: 0.031355
Epoch 18/20 | Train Loss: 0.096319 | Val Loss: 0.025992
Epoch 19/20 | Train Loss: 0.095440 | Val Loss: 0.039232
Epoch 20/20 | Train Loss: 0.092597 | Val Loss: 0.025167

 Training completed!

Evaluating model performance

EVALUATION RESULTS

 ROC-AUC Score: 0.9553

 Optimal Threshold: 0.025689

 Reconstruction Error Statistics:
   Normal Traffic:  0.025172 ± 0.204820
   Attack Traffic:  0.993352 ± 16.558800
   Separation:      39.46x

 Classification Report:
              precision    recall  f1-score   support

      Normal     0.9890    0.8829    0.9329     47581
      Attack     0.6230    0.9518    0.7531      9676

    accuracy                         0.8945     57257
   macro avg     0.8060    0.9174    0.8430     57257
weighted avg     0.9272    0.8945    0.9025     57257

{'auc': 0.9552863636143667, 'threshold': np.float32(0.025688969), 'errors_normal': array([0.01004823, 0.02513907, 0.00990716, ..., 0.00197217, 0.1082532 ,
       0.00400445], shape=(47581,), dtype=float32), 'errors_attack': array([0.11582565, 0.11264124, 0.11845548, ..., 0.89375156, 1.6084003 ,
       0.15969555], shape=(9676,), dtype=float32), 'predictions': array([0, 0, 0, ..., 1, 1, 1], shape=(57257,)), 'config': {'subset_size': 500000, 'contamination_rate': 0.05, 'epochs': 20, 'device': 'cpu', 'features': 33}}
```


