import numpy as np
import tkinter as tk
from tkinter import Frame, Button, Canvas
from scipy.ndimage import center_of_mass, shift

theme1 = "#121212"  
theme2 = "#99E5FF"  

class DigitRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")
        self.root.geometry("520x570")
        self.root.configure(bg=theme1)

        self.gridSize = 28
        self.canvasSize = 400
        self.cellSize = self.canvasSize // self.gridSize
        self.penThickness = 1

        self.grid = np.zeros((self.gridSize, self.gridSize), dtype=np.float32)
        self.setupUI()
        self.loadModel()
        self.isDrawing = False

    def setupUI(self):
        self.resultLabel = tk.Label(
            self.root, text="Draw a digit", font=("Segoe UI", 22, "bold"),
            bg=theme1, fg="white", pady=10
        )
        self.resultLabel.pack()

        self.canvas = Canvas(
            self.root, width=self.canvasSize, height=self.canvasSize,
            bg="black", highlightthickness=2, highlightbackground=theme2
        )
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self.startDrawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stopDrawing)

        buttonFrame = Frame(self.root, bg=theme1)
        buttonFrame.pack(pady=8)

        style = {
            "width": 20,
            "height": 2,
            "bg": "#1F1F1F",
            "fg": "white",
            "activebackground": "#333333",
            "activeforeground": theme2,
            "font": ("Segoe UI", 10, "bold"),
            "relief": "flat",
            "bd": 0
        }

        Button(buttonFrame, text="Clear", command=self.clearCanvas, **style).pack(side="left", padx=10)
        Button(buttonFrame, text="Recognize", command=self.recognizeDigit, **style).pack(side="right", padx=10)

    def loadModel(self):
        try:
            self.W1 = np.loadtxt("W1.txt")
            self.W2 = np.loadtxt("W2.txt")
            self.b1 = np.loadtxt("b1.txt").reshape(-1, 1)
            self.b2 = np.loadtxt("b2.txt").reshape(-1, 1)
            self.mean = np.loadtxt("mean.txt").reshape(-1, 1)
            self.std = np.loadtxt("std.txt").reshape(-1, 1)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.root.after(0, self.root.destroy)

    def startDrawing(self, event):
        self.isDrawing = True
        self.draw(event)

    def draw(self, event):
        if not self.isDrawing:
            return
        gridX = event.x // self.cellSize
        gridY = event.y // self.cellSize
        radius = self.penThickness

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = gridX + dx, gridY + dy
                if 0 <= nx < self.gridSize and 0 <= ny < self.gridSize:
                    dist_sq = dx**2 + dy**2
                    intensity = 1 / (1 + dist_sq) if dist_sq > 0 else 1.0
                    self.grid[ny, nx] = min(1.0, self.grid[ny, nx] + intensity)
        self.redrawCanvas()

    def stopDrawing(self, event):
        self.isDrawing = False

    def clearCanvas(self):
        self.grid = np.zeros((self.gridSize, self.gridSize), dtype=np.float32)
        self.canvas.delete("all")
        self.resultLabel.config(text="Draw a digit")

    def redrawCanvas(self):
        self.canvas.delete("all")
        for y in range(self.gridSize):
            for x in range(self.gridSize):
                value = self.grid[y, x]
                if value > 0:
                    brightness = int(min(255, value * 255))
                    color = f"#{brightness:02x}{brightness:02x}{brightness:02x}"
                    x1 = x * self.cellSize
                    y1 = y * self.cellSize
                    x2 = x1 + self.cellSize
                    y2 = y1 + self.cellSize
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=color)

    def centerAndDenoise(self):
        img = np.copy(self.grid)
        com = center_of_mass(img)
        shift_x = int(self.gridSize // 2 - com[1])
        shift_y = int(self.gridSize // 2 - com[0])
        centered = shift(img, shift=(shift_y, shift_x), mode='constant', cval=0.0)
        centered[centered < 0.1] = 0
        return centered

    def preprocessImage(self):
        img = self.centerAndDenoise()
        X = img.flatten().reshape(-1, 1) * 255.0
        X = (X - self.mean) / self.std
        return X

    def recognizeDigit(self):
        if np.sum(self.grid) == 0:
            self.resultLabel.config(text="Draw something!")
            return
        X = self.preprocessImage()
        prediction = self.forwardPass(X)
        self.resultLabel.config(text=str(prediction))

    def forwardPass(self, X):
        relu = lambda x: np.maximum(0, x)
        softmax = lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
        Z1 = self.W1 @ X + self.b1
        A1 = relu(Z1)
        Z2 = self.W2 @ A1 + self.b2
        A2 = softmax(Z2)
        return np.argmax(A2, axis=0)[0]

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognitionApp(root)
    app.run()
