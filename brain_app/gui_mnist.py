import PIL.Image, PIL.ImageTk
from tkinter import Tk, Button, Canvas, NW,Label,messagebox
from tkinter import filedialog
import numpy as np
import nibabel as nib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
root = Tk()
root.title("Title!!!!")
root.iconbitmap("brain.ico")
# root.geometry("800x400")

img = PIL.Image.open("hust.png")
# img = img.resize((300,400))
photo = PIL.ImageTk.PhotoImage(img)
imgDeco = Label(root, image = photo ).pack()


canvas = Canvas(root, width=200, height=200)
canvas.pack()
def browseCsv():
    global img
    filename = filedialog.askopenfile(parent=root,mode='rb',title='Choose a image')
    img = PIL.Image.open(filename)
    photo = PIL.ImageTk.PhotoImage(img.resize((200,200)))
    root.photo = photo
    canvas.create_image(0, 0, anchor=NW, image=photo)
def process(inputs, input_size = (28,28)):
    output = inputs.resize(input_size)
    img_arr = img_to_array(output)
    img_arr = img_arr[np.newaxis, ..., 0:1] / 255.0
    model = load_model("model_mnist.h5")
    res = np.argmax(model(img_arr))
    return res
label = None
def mess():
    global label
    if label :
        label.destroy()
    label = None
    res = process(img)
    result = "The number is : " +str(res)
    messagebox.showinfo("Result",result)
    label = Label(root, text = result)
    label.pack()

browseButton = Button(root, text = "Browse",padx = 20, pady = 10,
                     command=browseCsv, fg = "white",bg="black").pack(pady=10)
resButton = Button(root, text = "Result",padx = 25,pady = 10,
                     command=mess, fg = "white",bg="black").pack()


root.mainloop()

