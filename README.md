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
 
FGSM     epsilon=0.15  Fooling Rate=60%

PGD      epsilon=0.15  alpha=0.01875  steps=20  Fooling Rate=100%

## Requirements

* Python 3.10+
* TensorFlow 
* OpenCV
* Matplotlib
* NumPy
* Jupyter Notebook

## References

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.
- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE international conference on computer vision (pp. 618-626).
- Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572.
- Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017). Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083.

```
