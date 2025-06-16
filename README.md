# whisper-emotion-classifier

**Emotion classification from speech using Whisper encoder + MLP head**  
The goal of the project is to fine-tune the model based on encoder of OpenAI's Whisper to classify emotions in Russian speech and to evaluate its quality using Monte Carlo methods.

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
uv venv --python 3.11
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

This will open a local Gradio interface for real-time emotion recognition from audio or video (`.mp3`, `.mp4` and `.wav` formats are supported).

---

## 📊 Logging & Tracking with Weights & Biases (W&B)

During training, the model logs metrics and visualizations to [Weights & Biases](https://wandb.ai/):

- Training & validation loss
- Accuracy per epoch
- Confusion matrix & misclassification analysis

To enable logging, make sure you are logged in:

```bash
wandb login
```

By default, W&B is integrated in the training script:

```python
import wandb
wandb.init(project="whisper-emotion")
```

You can monitor training in real time at [wandb.ai](https://wandb.ai/) or in your terminal.

---

## 🧪 Experiments & Analysis

We performed uncertainty estimation using:

- `MC_dropout.ipynb`: Monte Carlo Dropout for evaluating prediction confidence
- `bootstrap.ipynb`: Bootstrap analysis for statistical variability
- `preprocess_data.ipynb`: Preprocessing steps — label mapping, splitting, etc.

---

## 📫 Contact

For questions or collaboration ideas, feel free to reach out via GitHub Issues.

---

> Made with ❤️ for movie lovers and ML enthusiasts
