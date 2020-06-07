from tkinter import *
from tkinter import filedialog
import tkinter as tk
# import tkinter.messagebox
# from video_cut import *
from tkinter.filedialog import *
from PIL import Image, ImageTk
from BestMomentMain import main


def process():
    E1.delete(0, END)
    filename = filedialog.askopenfilename(initialdir = "/", title = "Select A File", filetypes = (("mp4","*.mp4"),("All files","*.*")))
    if filename != "":
        path = main("./known_images", filename, "./data/", 1, 0.6, True)
        E1.insert(0, path)


top = tk.Tk()
bgcolor="brown4"
top.configure(bg=bgcolor)
top.title("Best Moment")
top.geometry('750x400')


L1 = tk.Label(top,text="WeLcOmE to bEsT mOmEnT",font=("Arial",40),bg=bgcolor,fg="chocolate1")
L1.grid(row=0,column=0,columnspan=2)

L2_s = "Wanna see your friends' hilarious moments?"
L3_s = "Click the button below to upload your video!"
L3_s2 = "and get your friend's best moment!"

L2 = tk.Label(top,text=L2_s,font=("Arial Bold",15),bg=bgcolor,fg="white")
L3 = tk.Label(top,text=L3_s,font=("Arial Bold",15),bg=bgcolor,fg="white")
L2.grid(row=1,column=0)
L3.grid(row=2,column=0)

B2 = tk.Button(top, text = "Click here to select a video~",font=("Arial Bold",15), bg="salmon1", command = process, fg="red")
B2.grid(row=3,column=0)

E1 = tk.Entry(top, bd=2, width="50")
E1.grid(row=4,column=0)

image = Image.open("emoji.jpg")
photo = ImageTk.PhotoImage(image)

label = Label(image=photo)
label.image = photo 
label.grid(row=1,rowspan=5,column=1,sticky=E)

top.mainloop()
