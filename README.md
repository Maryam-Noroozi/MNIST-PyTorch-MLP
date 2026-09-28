# MNIST-PyTorch-MLP
End-to-end MNIST digit classification using an Advanced MLP architecture in PyTorch.
# MNIST Digit Classification using PyTorch

In this project, I built a Multi-Layer Perceptron (MLP) from scratch using PyTorch to classify handwritten digits from the MNIST dataset. The goal was to establish a clean, standard PyTorch workflow from data pipeline setup to training and evaluation.

## 🚀 Results
* **Test Accuracy:** 97.25% after 5 epochs
* **Optimizer:** Adam (lr=0.001)
* **Loss Function:** CrossEntropyLoss

---

## 🏗️ Model Architecture

The model reshapes the $28 \times 28$ input images into a 784-element vector and passes them through fully connected layers with Dropout for regularization:
Input (1x28x28)
│
▼
Flatten (784)
│
▼
Linear (784 -> 128) ──> ReLU ──> Dropout (0.2)
│
▼
Linear (128 -> 64)  ──> ReLU
│
▼
Linear (64 -> 10)   ──> Output (Logits)
---

## 🛠️ Project Structure & Workflow

* `train.py`: Contains the complete data loading pipeline, network definition, standard 4-step PyTorch training loop, and evaluation script.
* `mnist_mlp_weights.pth`: Saved model weights for quick inference without retraining.

---

## 💻 How to Run

1. **Install Dependencies:**
   ```bash
   pip install torch torchvision
   ```

2. Train the model

  ```Bash
  python train.py
  ```

3. Load Saved Weights:

  ```python
  import torch
  from train import AdvancedMLP

  model = AdvancedMLP()
  model.load_state_dict(torch.load("mnist_mlp_weights.pth"))
  model.eval()
  ```

