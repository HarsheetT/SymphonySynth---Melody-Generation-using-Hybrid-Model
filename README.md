# SymphonySynth: Hybrid AI Melody Generation

SymphonySynth is an end-to-end generative AI framework designed to model and generate long-range melodic context. By fusing standard attention blocks with a structured state-space model, the network successfully mitigates the quadratic computational complexity of standard Transformers while preserving strict structural dependencies across extended sequence lengths.

This project was selected and presented live at the **Intech '25 Showcase**.

---

## 🚀 Core Features

* **Hybrid Architecture:** Fuses PyTorch `nn.TransformerEncoder` blocks with a custom **gated Mamba layer** (consisting of a 1D-CNN and a sigmoid gating mechanism) to model contextual musical dependencies across more than 5,000 tracks.
* **Symbolic Tokenization Pipeline:** Utilizes a custom **REMI-M tokenizer** to normalize musical tempos and map symbolic MIDI data into dense token sequences spanning up to 24,000+ tokens per file.
* **Targeted Domain Adaptation:** Features a multi-stage training routine, combining broad-spectrum pre-training on Western musical datasets with specialized domain fine-tuning on a curated Indian Raga dataset.
* **Interactive Inference Interface:** Deployed as an interactive, web-based application built on Streamlit using a REST-integrated backend to support real-time user image uploads, audio playback, and on-the-fly melody generation.

---

## 📊 Architecture & Model Performance

The hybrid design achieves excellent convergence metrics across sequential training environments by treating musical sequences as dense states. Categorical cross-entropy loss dropped substantially during execution:

| Training Phase | Target Dataset | Scale / Scope | Training Duration | Cross-Entropy Loss |
| :--- | :--- | :--- | :--- | :--- |
| **Base Pre-training** | Western Dataset | 5,000+ Tracks | 10 Epochs | `5.2` $\rightarrow$ `3.1` |
| **Domain Adaptation** | Indian Raga Dataset | Curated Collection | 3 Epochs | `3.1` $\rightarrow$ `1.95` |

---

## 📁 Repository Structure

```text
symphonysynth/
│
├── src/
│   ├── tokenizer.py        # Custom REMI-M tokenizer and processing pipeline
│   ├── model.py            # PyTorch Hybrid Transformer-Mamba neural network
│   ├── train.py            # Base training and domain fine-tuning logic
│   └── app.py              # REST API backend and generation core
│
├── interface/
│   └── web_app.py          # Interactive Streamlit frontend UI
│
├── data/
│   ├── raw_midi/           # Source Western and Indian Raga MIDI files
│   └── tokenized/          # Preprocessed token sequences
│
└── requirements.txt        # Core dependencies (Torch, Streamlit, NumPy, etc.)
```

## ⚡ Quick Start

### 1. Prerequisites & Installation
* **Description:** Clone this repository and set up your local development environment.
* **Command:**
```bash
  git clone [https://github.com/HarsheetT/SymphonySynth.git](https://github.com/HarsheetT/SymphonySynth.git)
  cd SymphonySynth
  pip install -r requirements.txt
```

### 2. Tokenize Dataset
To convert raw symbolic MIDI tracks into dense sequential token structures optimized for the hybrid network:
```bash
python src/tokenizer.py --input_dir data/raw_midi/ --output_dir data/tokenized/
```

### 3. Run Training & Fine-Tuning
Execute the multi-stage training pipeline (Base Pre-training followed by targeted Domain Adaptation):
```bash
##### Phase 1: Base Pre-training on the Western Dataset
python src/train.py --mode base --epochs 10

##### Phase 2: Domain-Specific Fine-Tuning on the Indian Raga Dataset
python src/train.py --mode finetune --epochs 3
```

### 4. Deploy the Local Web App
Launch the interactive Streamlit user interface to generate original, context-aware melodies in real time:
```bash
streamlit run interface/web_app.py
```


