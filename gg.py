from keras.models import load_model
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tkinter import *
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import pyscreenshot as ImageGrab
import cv2
model = load_model('CNN2.h5')

def predict_digit(img):
    img_array = np.asarray(img)
    resized = cv2.resize(img_array, (28, 28))
    gray_scale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)  # (28, 28)
    image = cv2.bitwise_not(gray_scale)
    image = image / 255.0
    image = image.reshape(1, 28, 28, 1)
    prediction = model.predict([image][0])
    return np.argmax(prediction), np.amax(prediction)

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("GUI Recognizer")
        self.x = self.y = 0
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="----Thinking----", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Recognise", command=self.classifyHandwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clearAll)
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        self.canvas.bind("<B1-Motion>", self.drawLines)

    # enables the canvas to draw oval dots to draw a digit
    def drawLines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')

    # clears the canvas if the clear button is clicked
    def clearAll(self):
        self.canvas.delete("all")

    # display the image
    def imageDisplay(imageNmber):
        fig, ax = plt.subplots()
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        # Plot the values of a 2D matrix or array as color-coded image
        ax.matshow(digit_matrix.images[image_number], cmap=plt.cm.binary)

    # classifies the handwritten digit
    def classifyHandwriting(self):
        HWND = self.canvas.winfo_id()  # get the handle of the canvas
        rect = self.canvas.coords(HWND)  # get the coordinate of the canvas
        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        im = ImageGrab.grab(rect).crop((x, y, x1, y1))
        digit, acc = predict_digit(im)
        self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')

app = App()
mainloop()