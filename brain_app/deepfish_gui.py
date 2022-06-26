import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.constants import HORIZONTAL
import numpy as np
from PIL import Image, ImageTk
import nibabel as nib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import array_to_img
#from utils import *
from haven import haven_chk as hc
from haven import haven_results as hr
from haven import haven_utils as hu
import torch
import torchvision
import tqdm
import pandas as pd
import pprint
import itertools
import sis, os
import pylab as plt
import time
import numpy as np

from src import *
import argparse

from torch.utils.data import sampler
from torch.utils.data.sampler import RandomSampler
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models
from torchinfo import summary
import torchvision.transforms as T
import exp_configs

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)
import matplotlib.pyplot as plt

import torch
from torch.backends import cudnn
from torch import nn

SIZE_OF_CANVAS_WiDTH = 512
SIZE_OF_CANVAS_HEIGH = 512


root = tk.Tk()
message = tk.messagebox.showinfo("Notification","you need to ...")
root.title("deepfish_gui")
root.iconbitmap("brain.ico")
img = Image.open("hust.png")
img = img.resize((512,512))
photo = ImageTk.PhotoImage(img)
imgDeco = tk.Label(root, image = photo ).grid(row=0, column=0, columnspan=4, rowspan=3)
#model = load_model("model_dense.h5", custom_objects={"Swish":Swish}, compile=False)

# Create model, opt, wrapper
model = models.get_model(exp_dict["model"], exp_dict=exp_dict).cuda()
opt = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.0005)
#model = wrappers.get_wrapper(exp_dict["wrapper"], model=model_original, opt=opt).cuda()
#if args.exp_config == 'loc':
    #batch = torch.utils.data.dataloader.default_collate([train_set[3]])
#else:
    #batch = torch.utils.data.dataloader.default_collate([train_set[0]])

#volumeT1 = np.zeros((256,256,256,1))
#mask3D = np.zeros((256,256,256,1))
def browseCsv():
    #global volumeT1, volumeT2, patchT1, patchT2
    filename = filedialog.askopenfile(parent=root,mode='rb',title='Choose image weight')
    filename = filename.name
    volume = nib.load(filename)
    '''if "T1" in filename:
        volumeT1 = nib.load(filename).get_data().astype("uint16")
        volumeT2 = nib.load(filename.replace("T1", "T2")).get_data().astype("uint16")
    elif "T2" in filename:
        volumeT2 = nib.load(filename).get_data().astype("uint16")
        volumeT1 = nib.load(filename.replace("T2", "T1")).get_data().astype("uint16")
    else : 
        tk.messagebox.showerror("Error","please choose an image T1W or T2W")
    preprocessedVolumeT1 = normalize(volumeT1)
    preprocessedVolumeT2 = normalize(volumeT2)
    patchT1 = extract_patches(preprocessedVolumeT1, PATCH_SIZE, EXTRACTION_STEP)
    patchT2 = extract_patches(preprocessedVolumeT2, PATCH_SIZE, EXTRACTION_STEP)'''
    
def getIndexButton1(index0):
    index0 = int(index0)
    if index0 < volumeT1.shape[0]:
        array_image = volumeT1[index0,...,0]
        canvasX.image =  ImageTk.PhotoImage(image=array_to_img(array_image[...,np.newaxis]))
        canvasX.create_image(0,0, anchor="nw", image=canvasX.image)
        
        array_mask = mask3D[index0,...,0]
        rgb_mask = gray2rgb(array_mask)
        canvas_maskX.image =  ImageTk.PhotoImage(image=array_to_img(rgb_mask))
        canvas_maskX.create_image(0,0, anchor="nw", image=canvas_maskX.image)

def getIndexButton2(index1):
    index1 = int(index1)
    if index1 < volumeT1.shape[1]:
        array_image = volumeT1[:,index1,:,0]
        canvasY.image =  ImageTk.PhotoImage(image=array_to_img(array_image[...,np.newaxis]))
        canvasY.create_image(0,0, anchor="nw", image=canvasY.image)
        
        array_mask = mask3D[:,index1,:,0]
        rgb_mask = gray2rgb(array_mask)
        canvas_maskY.image =  ImageTk.PhotoImage(image=array_to_img(rgb_mask))
        canvas_maskY.create_image(0,0, anchor="nw", image=canvas_maskY.image)

def getIndexButton3(index2):
    index2 = int(index2)
    if index2 < volumeT1.shape[2]:
        array_image = volumeT1[:,:,index2,0]
        canvasZ.image =  ImageTk.PhotoImage(image=array_to_img(array_image[...,np.newaxis]))
        canvasZ.create_image(0,0, anchor="nw", image=canvasZ.image)
        
        array_mask = mask3D[:,:,index2,0]
        rgb_mask = gray2rgb(array_mask)
        canvas_maskZ.image =  ImageTk.PhotoImage(image=array_to_img(rgb_mask))
        canvas_maskZ.create_image(0,0, anchor="nw", image=canvas_maskZ.image)
        
def processImage():
    try : 
        global patchT1, patchT2
        pred = np.zeros((*patchT1.shape[:-1], NUM_CLASS))
        valid_index = np.where(np.sum(patchT1, axis=(1, 2, 3, 4)) != 0)
        pred[valid_index] = model.predict({"inputT1": patchT1[valid_index],
                                           "inputT2": patchT2[valid_index]}, batch_size = 4)[0]
        label_constructed = reconstruct_volume(pred, volumeT1.shape[:-1], EXTRACTION_STEP)
        output_standard = np.expand_dims(np.argmax(label_constructed, axis=-1), axis=-1).astype(np.uint8)

        del patchT1, patchT2, pred, label_constructed,valid_index
        return output_standard
    except:
        global mask3D
        return mask3D

def mess():
    global mask3D
    printout = tk.Label(root)
    mask3D = processImage()
    printout.grid(row=9, column=1)
    printout.configure(text = "Segmented done!!!")



canvasX = tk.Canvas(root, width=SIZE_OF_CANVAS_WiDTH,height=SIZE_OF_CANVAS_HEIGH)
canvasX.grid(row=5, column=0)
canvasY = tk.Canvas(root,width=SIZE_OF_CANVAS_WiDTH,height=SIZE_OF_CANVAS_HEIGH)
canvasY.grid(row=5, column=1)
canvasZ = tk.Canvas(root,width=SIZE_OF_CANVAS_WiDTH,height=SIZE_OF_CANVAS_HEIGH)
canvasZ.grid(row=5, column=2)

canvas_maskX = tk.Canvas(root,width=SIZE_OF_CANVAS_WiDTH,height=SIZE_OF_CANVAS_HEIGH)
canvas_maskX.grid(row=10, column=0)
canvas_maskY = tk.Canvas(root,width=SIZE_OF_CANVAS_WiDTH,height=SIZE_OF_CANVAS_HEIGH)
canvas_maskY.grid(row=10, column=1)
canvas_maskZ = tk.Canvas(root,width=SIZE_OF_CANVAS_WiDTH,height=SIZE_OF_CANVAS_HEIGH)
canvas_maskZ.grid(row=10, column=2)

scaleButton1 = tk.Scale(root, from_=0, to = 255, orient= HORIZONTAL,length=200,
                        command=getIndexButton1)
scaleButton1.grid(row=4, column=0)
scaleButton2 = tk.Scale(root, from_=0, to = 255, orient= HORIZONTAL,length=200, 
                        command=getIndexButton2)
scaleButton2.grid(row=4, column=1)
scaleButton3 = tk.Scale(root, from_=0, to = 255, orient= HORIZONTAL,length=200,
                        command=getIndexButton3)
scaleButton3.grid(row=4, column=2)
browseButton = tk.Button(root, text = "Browse",padx = 20,pady = 10,
                         command=browseCsv,fg = "white",bg="black")
browseButton.grid(row=7, column=1)
resButton = tk.Button(root, text = "Result",padx = 25,pady = 10,
                     command=mess, fg = "black",bg="white")
resButton.grid(row=8, column=1)

label1 = tk.Label(text = "x-axis")
label1.grid(row=3, column=0)
label2 = tk.Label(text = "y-axis")
label2.grid(row=3, column=1)
label3 = tk.Label(text = "z-axis")
label3.grid(row=3, column=2)
root.mainloop()


