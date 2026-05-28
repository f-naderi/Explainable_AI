```markdown
# MNIST Handwritten Digit Recognition

A deep learning project for recognizing handwritten digits (0-9) using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

## Features

- CNN model with 99%+ accuracy on MNIST test set
- Grad-CAM visualization for model explainability
- Adversarial attack generation (FGSM & PGD)
- GPU support via WSL2

## Project Structure

Explainable_AI/
├── model.py            # Model architecture (CNN)
├── utils.py            # Helper functions, Grad-CAM, data loading
├── train.py            # Training script
├── evaluate.py         # Evaluation with XAI (Grad-CAM)
├── predict.py          # Single image prediction
├── attacks.py          # FGSM & PGD adversarial attacks
├── MNIST.ipynb         # Interactive Jupyter notebook
└── requirements.txt    # Python dependencies

## Installation

```bash
git clone https://github.com/f-naderi/Explainable_AI.git
cd Explainable_AI
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Train the model

```bash
python3 train.py
```

### Evaluate with Grad-CAM

```bash
python3 evaluate.py
```

### Generate adversarial examples

```bash
python3 attacks.py
```

## Results

Model Test Accuracy
CNN 99.05%

Adversarial Attacks

Attack 
FGSM     epsilon=0.15  Fooling Rate=60%
PGD      epsilon=0.15  alpha=0.01875  steps=20  Fooling Rate=100%

## Requirements

· Python 3.10+
· TensorFlow 
· OpenCV
· Matplotlib
· NumPy

## References

- MNIST Dataset: LeCun et al. (1998)
- Grad-CAM: Selvaraju et al. (2017)
- FGSM Attack: Goodfellow et al. (2015)
- PGD Attack: Madry et al. (2018)

```
