# GPU-Powered AI Framework for Zero-Day Threat Prediction

## Overview
This project builds a GPU-accelerated AI system that detects and predicts **zero-day cyber threats** — attacks that exploit unknown vulnerabilities.  
Using deep learning (Transformers) and TACC GPUs, the system learns normal network behavior and flags unseen anomalies in large-scale traffic datasets.

---

## Project Workflow

1️⃣ Data Acquisition → 2️⃣ Data Preprocessing → 3️⃣ Model Development  
→ 4️⃣ GPU Training → 5️⃣ Evaluation → 6️⃣ Explainability → 7️⃣ Deployment → 8️⃣ Visualization

---

## Step 1: Data Acquisition

###  Primary Datasets
#### 1. [CICIDS2017 – Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/cicids2017.html)
- Includes normal & 14 attack types (DDoS, Botnet, PortScan, etc.)
- Format: PCAP + CSV  
- Alternative mirror: [Kaggle Link](https://www.kaggle.com/datasets/cicdataset/cicids2017)

#### 2. [UNSW-NB15 – Australian Cyber Security Centre](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- Contains 49 features + 9 attack types
- CSV and PCAP files  
- Mirror: [Kaggle Link](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)

#### 3. [MAWILab Dataset (Optional)](http://www.fukuda-lab.org/mawilab/)
- Real backbone network traces with anomaly labels
- Ideal for robustness testing

###  Supporting Datasets (Optional)
- [CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html)
- [Bot-IoT Dataset](https://research.unsw.edu.au/projects/bot-iot-dataset)
- [TON_IoT](https://research.unsw.edu.au/projects/toniot-datasets)

###  Download Method
1. Visit the official dataset link.
2. Fill out the academic download form.
3. Receive the download link by email (Google Drive / FTP).
4. Download `.csv` or `.pcap` files to your local machine.

### ⬆️ Upload to TACC
```bash
scp -r ~/Downloads/CICIDS2017 your_tacc_username@frontera.tacc.utexas.edu:/home1/your_tacc_username/zero_day_gpu/data/
```

---

## ⚙️ Step 2: Environment & Dependencies

### 🔧 Hardware
- NVIDIA GPU (A100/V100/T4)
- 32–64 GB RAM
- 100 GB storage
- TACC account access (Lonestar6 / Frontera)

### Software Stack
| Category | Tool |
|-----------|------|
| Language | Python 3.10 + |
| Framework | PyTorch 2.1 / TensorFlow 2.15 |
| GPU Stack | CUDA 12.2 + cuDNN 8.x |
| Data | RAPIDS cuDF, cuML, Pandas |
| Visualization | Streamlit, Matplotlib |
| Explainability | SHAP, LIME, Captum |
| Deployment | TorchServe / Triton |
| Scheduler | SLURM (TACC) |

###  Installation (Conda)
```bash
conda create -n zeroday_ai python=3.10
conda activate zeroday_ai
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c rapidsai -c nvidia -c conda-forge -c defaults cudf=23.08 python=3.10
pip install scikit-learn shap lime streamlit seaborn matplotlib
```

Verify GPU:
```python
import torch
print(torch.cuda.is_available())
```

---

##  Step 3: Data Preprocessing
```python
import cudf
df = cudf.read_csv("data/CICIDS2017.csv")
df = df.dropna()
df = df.sample(frac=0.1)  # sample for faster testing
```

- Normalize numeric features  
- Encode categorical fields (`protocol`, `service`, etc.)  
- Split into training / testing sets

---

##  Step 4: Model Development
Example Transformer Autoencoder:
```python
import torch, torch.nn as nn

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim=64, nhead=4):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead), num_layers=3)
        self.decoder = nn.Linear(input_dim, input_dim)
    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
```

---

##  Step 5: GPU Training on TACC

**Job script (`train_job.sh`):**
```bash
#!/bin/bash
#SBATCH -J ZeroDayAI
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH -t 02:00:00
module load python/3.10 cuda/12.2 pytorch/2.1.0
python train.py
```
Submit with:
```bash
sbatch train_job.sh
```

---

## Step 6: Model Evaluation
```python
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))
```
Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC, Latency.

---

##  Step 7: Explainability
```python
import shap
explainer = shap.Explainer(model, sample_data)
shap_values = explainer(sample_data)
shap.plots.beeswarm(shap_values)
```

---

##  Step 8: Deployment & Visualization
**Serve model:**
```bash
torch-model-archiver --model-name zeroday --version 1.0 --serialized-file model.pt --handler handler.py
torchserve --start --ncs --model-store model_store --models zeroday=zeroday.mar
```

**Build dashboard:**
```bash
streamlit run dashboard.py
```

Dashboard displays:
- Live anomaly scores  
- Feature importance (SHAP)  
- Attack type trends over time  

---

##  Folder Structure
```
zero_day_gpu/
├── data/
│   ├── CICIDS2017.csv
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
├── models/
│   ├── transformer_autoencoder.pt
├── dashboard/
│   ├── dashboard.py
├── train_job.sh
├── PROJECT_GUIDE.md
└── requirements.txt
```

---

##  Step 9: Documentation
Include in your final report:
- Project background  
- Dataset & preprocessing  
- Model architecture & training  
- GPU performance comparison  
- Explainability visualizations  
- Results & conclusions  

---

##  Summary
This project integrates **AI + GPU computing** to build a scalable, explainable, and adaptive cybersecurity system capable of identifying **zero-day threats** before they strike.  
Using open datasets, PyTorch Transformers, and TACC GPUs, it demonstrates the future of **AI-driven cyber defense**.
