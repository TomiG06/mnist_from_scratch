import numpy as np
from datasets import load_dataset
from tqdm import trange
import matplotlib.pyplot as plt

def load_mnist():
    data = load_dataset("mnist")

    X_train = np.array([np.array(img) for img in data["train"]["image"]]).astype(np.float32) / 255
    Y_train = np.array(data["train"]["label"])
    X_test  = np.array([np.array(img) for img in data["test"]["image"]]).astype(np.float32) / 255
    Y_test = np.array(data["test"]["label"])

    return X_train, Y_train, X_test, Y_test


# TODO: add biases
# better structure, probably a 

def Linear(in_dim, out_dim):
    return (np.random.randn(in_dim, out_dim) * np.sqrt(2/in_dim)).astype(np.float32)

def ReLU(x):
    return np.maximum(0, x)

def Softmax(z):
    if z.ndim == 1:
        # Single input
        axis = 0
        keepdims = False
    else:
        # Batch
        axis = 1
        keepdims = True

    exps = np.exp(z - np.max(z, axis=axis, keepdims=keepdims))
    return exps/exps.sum(axis=axis, keepdims=keepdims)

# We are using one-hot encoding so the non-true zero out
def CrossEntropyLoss(y_out, truths):
    # we use epsilon to preven any log(0) cases
    EPSILON = 1e-9
    y_out_correct = y_out[np.arange(len(truths)), truths]
    return np.mean(-np.log(y_out_correct + EPSILON))


if __name__ == "__main__":
    print("Preparing MNIST dataset (this might take a while)")
    X_train, Y_train, X_test, Y_test = load_mnist()

    l1 = Linear(28*28, 128)
    l2 = Linear(128, 10)

    v1 = np.zeros_like(l1)
    v2 = np.zeros_like(l2)

    EPOCHS = 25
    BATCH_SIZE = 64
    LR = 0.01
    LR_DECAY = np.log(2) / 10
    MOMENTUM = 0.9

    losses, accuracies = [], []

    print("Starting Training")
    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch+1}/{EPOCHS}")
        t = trange(len(X_train) // BATCH_SIZE)

        # Learning Rate Exponential Decay
        lr = LR * np.exp(-LR_DECAY * epoch)

        for i in t:
            batch = np.arange(BATCH_SIZE * i, min(BATCH_SIZE * (i+1), len(X_train)))

            X = X_train[batch].reshape((-1, 28*28))
            Y = Y_train[batch] # classes

            # one-hot representation
            Y_one_hot = np.zeros((len(Y), 10))
            Y_one_hot[np.arange(len(Y)), Y] = 1

            # Forward Pass

            # Layer 1
            l1_out = X.dot(l1)

            # Layer 1 activation
            l1_out_relu = ReLU(l1_out)

            # Layer 2 
            l2_out = l1_out_relu.dot(l2)

            # Final activation
            sm = Softmax(l2_out)

            # Loss
            loss = CrossEntropyLoss(sm, Y)

            # Backward Pass (Calculating gradients)

            dLoss_dl2_out = 1/BATCH_SIZE * (sm - Y_one_hot)

            dl_2 = l1_out_relu.T.dot(dLoss_dl2_out)

            dLoss_drelu = dLoss_dl2_out.dot(l2.T)

            dLoss_dl1_out = dLoss_drelu * (l1_out_relu > 0)

            dl_1 = X.T.dot(dLoss_dl1_out)

            # SGD (with Momentum)
            v1 = MOMENTUM * v1 - lr * dl_1
            v2 = MOMENTUM * v2 - lr * dl_2

            l1 += v1
            l2 += v2

            losses.append(loss)
            accuracy = (np.argmax(sm, axis=1) == Y).astype(np.float32).mean()
            accuracies.append(accuracy)

            t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))


    plt.plot(losses)
    plt.plot(accuracies)
    plt.title("Loss vs Accuracy")
    plt.xlabel("Iterations")
    plt.show()

    print("Running Tests...")
    accuracies.clear()

    for img, lbl in zip(X_test, Y_test):
        z_1 = img.flatten().dot(l1)
        rel = ReLU(z_1)
        z_2 = rel.dot(l2)
        sm = Softmax(z_2)
        
        accuracies.append(int(np.argmax(sm) == lbl))

    print("Final Accuracy:", np.mean(accuracies))
