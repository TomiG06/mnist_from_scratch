# this thing is vibe coded so i guarantee nothing

import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
import os


class NN:
    def __init__(self, l1, l2):
        self.l1 = l1
        self.l2 = l2

    def forward(self, X):
        z1 = X.dot(self.l1)
        relu = np.maximum(z1, 0)
        z2 = relu.dot(self.l2)
        return NN.softmax(z2)

    @staticmethod
    def softmax(Y):
        exps = np.exp(Y - np.max(Y))
        return exps/exps.sum()


class UI:
    def __init__(self, root, model):
        self.model = model

        self.root = root
        self.root.title("UI Prediction")

        self.canvas_width = 280
        self.canvas_height = 280

        self.canvas = tk.Canvas(root, bg="black", width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw_image = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.draw)  # Hold left mouse button and move
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        self.frame = tk.Frame(root)
        self.frame.pack(padx=10, pady=10)

        # Button below the canvas
        self.clear_button = tk.Button(self.frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(pady=10)

        self.predict_button = tk.Button(self.frame, text="Predict", command=self.predict)
        self.predict_button.pack(pady=10)

        self.label_frame = tk.Frame(self.frame)
        self.label_frame.pack(pady=(10, 0))

        self.labels = []
        self.bars = []

        self.last_x, self.last_y = None, None

        for i in range(10):
            item_frame = tk.Frame(self.label_frame)
            item_frame.grid(row=0, column=i, padx=2)

            label = tk.Label(item_frame, text=str(i), width=4)
            label.pack()

            # Bar canvas
            bar = tk.Canvas(item_frame, width=20, height=50, bg="white", highlightthickness=0)
            bar.pack()

            # Draw a default bar (e.g., 30% height)
            bar_height = 30  # change this dynamically
            bar.create_rectangle(0, 50 - bar_height, 20, 50, fill="blue")

            self.labels.append(label)
            self.bars.append(bar)

        self.last_x, self.last_y = None, None

    def draw(self, event):
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    fill="white", width=20, capstyle=tk.ROUND, smooth=True)
            self.draw_image.line([self.last_x, self.last_y, event.x, event.y], fill=0, width=20)

        self.last_x, self.last_y = event.x, event.y

    def reset(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw_image = ImageDraw.Draw(self.image)

    def update_bars(self, values):  # values is a list of 10 floats in [0,1]
        for i, v in enumerate(values):
            self.bars[i].delete("all")
            h = int(50 * v)
            self.bars[i].create_rectangle(0, 50 - h, 20, 50, fill="blue")

    def get_canvas_array(self):
        img = self.image.resize((28, 28))        

        # Convert to NumPy array and normalize
        arr = np.asarray(img, dtype=np.uint8)
        arr = 1 - arr/255
        return arr.flatten()

    def predict(self):
        stats = self.model.forward(self.get_canvas_array().flatten())
        self.update_bars(stats)
    

if __name__ == "__main__":
    weights_filename = "mnist_from_scratch_weights.npz"
    if not os.path.exists(weights_filename):
        print(f"{weights_filename} does not exist")
        os.exit(1)

    weights = np.load(weights_filename)
    nn = NN(weights["l1"], weights["l2"])

    root = tk.Tk()
    app = UI(root, nn)
    root.mainloop()

