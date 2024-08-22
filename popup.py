import sys
import tkinter as tk
from tkinter import messagebox

def show_popup(result):
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Prediction Result", result)
    root.destroy()

if __name__ == '__main__':
    result = sys.argv[1]
    show_popup(result)    