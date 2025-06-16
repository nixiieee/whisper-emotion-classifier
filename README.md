# whisper-emotion-classifier

**Emotion classification from speech using Whisper encoder + MLP head**  
Built on top of OpenAI's Whisper architecture, this project fine-tunes the encoder to classify emotions in Russian speech.

---

## 📁 Project Structure

```
whisper-emotion-classifier/
│
├── inference/
│   ├── gradio_pipeline.py     # Gradio demo pipeline
│   └── inference.py           # Inference classes and logic
│
├── model/
│   └── train_small.py         # Whisper-small training script
│
├── preprocess_data.ipynb      # Data preprocessing pipeline
├── MC_dropout.ipynb           # MC Dropout experiments for uncertainty
├── bootstrap.ipynb            # Bootstrap analysis for uncertainty
├── requirements.txt           # List of dependencies
└── README.md
```

---

## 🤗 Hugging Face Resources

- Dataset: [`nixiieee/dusha_balanced`](https://huggingface.co/datasets/nixiieee/dusha_balanced)  
- Model: [`nixiieee/whisper-small-emotion-classifier-dusha`](https://huggingface.co/nixiieee/whisper-small-emotion-classifier-dusha)

---

## ⚙️ Model Architecture

The model consists of:

- **Encoder**: Pretrained Whisper encoder (`openai/whisper-small`)
- **Head**: Lightweight MLP for emotion classification (softmax output)
- Fine-tuned on emotional Russian speech (multi-class classification)

---

## 🚀 Quickstart

### 1. Install dependencies

#### Using [`uv`](https://github.com/astral-sh/uv) (recommended):

```bash
uv venv -p 3.11
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -r requirements.txt
```

#### Alternatively, using pip:

```bash
pip install -r requirements.txt
```

### 2. Launch Gradio App

```bash
python3 inference/gradio_pipeline.py
```

This will open a local Gradio interface for real-time emotion recognition from audio.

---

## 🧪 Experiments & Analysis

We performed uncertainty estimation using:

- 📘 `MC_dropout.ipynb`: Monte Carlo Dropout for evaluating prediction confidence
- 📗 `bootstrap.ipynb`: Bootstrap analysis for statistical variability
- 📙 `preprocess_data.ipynb`: Preprocessing steps — label mapping, splitting, etc.

---

## ✅ Requirements

Dependencies are listed in `requirements.txt`. Install with `uv` (recommended) or `pip`.
