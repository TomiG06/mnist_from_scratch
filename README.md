# MNIST from scratch

This repository implements a simple, 2-Layer Neural Network and trains it over the MNIST dataset, a classic collection of 70,000 handwritten digit images.

## Setup and Run

```
git clone https://github.com/TomiG06/mnist_from_scratch.git
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python mnist.py
```

To exit your virtual environment execute the `deactivate` command.

## Network Architecture

| Layer         | Input Size | Output Size | Activation |
|---------------|------------|-------------|------------|
| Linear Layer 1| 784        | 128         | ReLU       |
| Linear Layer 2| 128        | 10          | Softmax    |

## Training

The model is trained using 60,000 images from the MNIST dataset (out of the 70,000 images). Below are the key parameters and functions used for during training:

| Parameter   | Value |
|-------------|-------|
| EPOCHS      | 25    |
| BATCH_SIZE  | 64    |

### Weight Initialization

The network weights were initialized using Kaiming Initialization

### **Loss Function**: *Cross Entropy Loss*

### **Optimizer**: *Stochastic Gradient Descent (SGD)*

| Parameter           | Value       |
|---------------------|-------------|
| Learning Rate       | 0.01        |
| Learning Rate Decay | ln(2) / 10  |
| Decay Type          | Exponential |
| Momentum            | 0.9         |

## Testing

In order to evaluate the model's performance, we utilize the other 10,000 images of the MNIST dataset our model has never seen during training.

**Accuracy: ~ 97 %**


## Other

- After training, a plot appears showing the loss and accuracy over iterations
- There is the ability to store the weights as a `.npz` file

## Todo

- [ ] Biases
- [ ] L2 regularization
- [ ] Early stopping

