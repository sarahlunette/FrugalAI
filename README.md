# 🟢 FrugalAI

**Answers to the challenge FrugalAI task — Deforestation Detection**
This repository contains code, notebooks, and results related to the **Frugal AI Challenge** — an ML competition encouraging efficient and environmentally sustainable AI models (e.g., minimizing energy, CO₂ impact) while maintaining strong performance. ([GitHub][2])

---

## 📌 Project Overview

The goal of this repository is to **build a model to detect illegal deforestation (e.g., chainsaw sounds) in audio recordings** while prioritizing **frugality**: low compute, low energy, and high accuracy.

This includes:

* A **Jupyter notebook pipeline for model development**
* A `RandomForest.ipynb` showing a lightweight classical model
* Data preprocessing artifacts in `spectrograms/`
* Quantified results (accuracy and environmental metrics)

---

## 📁 Repository Structure

```
FrugalAI/
├── spectrograms/            # (likely) audio preprocessing outputs or utilities
├── FrugalAI.ipynb           # Main notebook: model development & evaluation
├── RandomForest.ipynb       # Notebook exploring Random Forest model baseline
├── README.md                # This comprehensive README
```

---

## 🧠 Challenge Context

This project was built as part of the **Frugal AI Challenge** — an initiative promoting efficient AI solutions that balance **performance with environmental sustainability**. Participants were evaluated not only on accuracy but also on energy consumption during training & inference. ([GitHub][2])

Your repository’s **FrugalAI.ipynb** likely includes loading audio data, transforming it into spectrograms/features, training a model, validating results, and reporting the energy footprint.

---

## 🚀 Quick Start

### 🛠️ Requirements

General tools you’ll need:

* **Python 3.8+**
* Jupyter Notebook or Jupyter Lab
* ML packages (e.g., scikit‑learn, librosa, numpy, pandas)
* Optional: visualization libraries (matplotlib, seaborn)

Install dependencies (example):

```bash
pip install numpy scipy pandas scikit-learn librosa matplotlib
```

> You may need to add additional packages based on the notebooks’ imports.

---

### 📊 Running the Notebook

1. **Open `FrugalAI.ipynb` in Jupyter**

   * Follow the cells: load data, preprocess audio, extract features.
   * Train and evaluate your model.
   * Record metrics such as accuracy, energy consumption, and CO₂ impact (if measured).

2. **Explore `RandomForest.ipynb`**

   * Compares a *Random Forest* baseline — often efficient and interpretable.
   * Typically useful for tabular or engineered audio feature data.

3. **Usage of `spectrograms/`**

   * Contains outputs from audio → spectrogram conversion for model input.
   * You can visualize, augment, or use these features for ML pipelines.

---

## 📈 Evaluation & Results

According to the repository summary:

* Your Random Forest baseline achieved **~80% accuracy**
* With extremely low energy usage: **≈ 0.00017 kg CO₂ and ≈ 0.003002 kWh** of electricity (measured or estimated) — demonstrating frugality. ([GitHub][3])

This aligns with the **“frugal” criteria** of the challenge: strong performance with minimal environmental and computational cost.

---

## 🧩 How It Works (High Level)

1. **Data Input**

   * Audio recordings (e.g., short clips of chainsaw vs. environment)
   * Converted to spectrograms (time‑frequency representations)

2. **Feature Extraction**

   * Extract spectral features or embed audio for classification

3. **Model Training**

   * Lightweight models (e.g., Random Forest, small neural networks)
   * Emphasis on low training and inference cost

4. **Evaluation**

   * Measure accuracy on validation/test sets
   * Optionally log energy impact (e.g., using packages like codecarbon)

5. **Reporting**

   * Document results, charts, and insights in notebooks

---

## 🧪 Example Code Snippet (Audio to Spectrogram)

Here’s a typical snippet you might see in the notebooks (example for feature extraction):

```python
import librosa
import numpy as np

def extract_spectrogram(audio_path, sr=12000, n_mels=128):
    y, sr = librosa.load(audio_path, sr=sr)
    spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    spect_db = librosa.power_to_db(spect, ref=np.max)
    return spect_db

spectrogram = extract_spectrogram("sample_audio.wav")
```

Use these spectrograms as features in any ML classifier.

---

## 📦 Dependencies (suggested)

Typical Python packages include:

* `numpy`
* `pandas`
* `scikit-learn`
* `librosa`
* `matplotlib` / `seaborn`
* `notebook`

Add these to a `requirements.txt` or Conda environment for reproducibility.

---

## 🏆 Challenge Impact

Frugal AI encourages **responsible and efficient AI** — prioritizing both **accuracy and sustainability**. Models like yours demonstrate how lightweight approaches like Random Forests or simple feature‑based classifiers can be competitive while drastically reducing energy consumption. ([GitHub][2])

---

## ❤️ Contributing

If this evolves further, consider adding:

* Dataset loader scripts
* A CLI or Python module for reproducible runs
* Scripts to measure and record environmental metrics
* A Dockerfile or Binder config for easy sharing
* More model baselines (light neural nets, boosted trees)

---

## 📄 License

If not already specified, consider adding an **MIT License** or another permissive open‑source license.

