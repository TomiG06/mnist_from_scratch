import numpy as np

class Conv:
    def __init__(self, in_c, out_c, kernel_size):
        self.kernel = np.random.uniform(size=(kernel_size, kernel_size))
        self.in_c = in_c
        self.out_c = out_c

    def forward(self, x):
        pass

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    c = Conv(1, 32, 5)

    images = np.load("images.npy").reshape((-1, 28, 28, 1))
    print(images[0])
