from tkinter import *
from tkinter import filedialog
import cv2
import os
import io
from PIL import Image, ImageTk
import numpy as np
from skimage.morphology import remove_small_objects, remove_small_holes
import tensorflow 
from tensorflow.keras.models import model_from_json
from tkinter import messagebox  
def net(path_arc,path_weights):
    json_file = open(path_arc, 'r')
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path_weights)
    return loaded_model

class FixedDropout(tensorflow.keras.layers.Dropout):
        def _get_noise_shape(self, inputs):
            if self.noise_shape is None:
                return self.noise_shape

            symbolic_shape = tensorflow.keras.backend.shape(inputs)
            noise_shape = [symbolic_shape[axis] if shape is None else shape
                           for axis, shape in enumerate(self.noise_shape)]
            return tuple(noise_shape)
def swish(x):
        """Swish activation function: x * sigmoid(x).
        Reference: [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
        """

        #if backend.backend() == 'tensorflow':
         #       return backend.tf.nn.swish(x)
        return x * tensorflow.keras.backend.sigmoid(x)
    #return  swish
def init():
  global classify_model
  global segment_model
  classify_model = tensorflow.keras.models.load_model(r"classify.hdf5",
    custom_objects= {"swish":swish, "FixedDropout":FixedDropout})
  segment_model = net(r"unet_segment_lung.json",r"segment_lung10_0.98.h5")
  print("model loaded!!!")
  return None
def full_process(image_test, thresh_classify = 0.5, thresh_segment = 0.5, border_pixel = 20, thresh_obj = 100, thresh_hole = 36):
  h,w = 192,288
  global result_mess
  if image_test.shape[-1] == 3:
    gray = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)
  im = cv2.resize(gray,(320,320))
  x_test =im.reshape((1,320,320,1))
  if image_test.shape[-1] == 1:
    x_test_real = np.concatenate((image_test, image_test, image_test), axis=-1)
    x_test_real = cv2.resize(x_test_real, (w, h))
    x_test_real = np.expand_dims(x_test_real, axis=0)
  elif image_test.shape[-1] == 3:
    x_test_real = cv2.resize(image_test, (w, h))
    x_test_real = np.expand_dims(x_test_real, axis=0)
  #print("here")
  #x_test = x_test_real[...,:1]
  y_preds = segment_model.predict(x_test)
  #cv2.imshow("ff",y_preds[0][:,:,0])
  #cv2.waitKey(0)
  y_preds = y_preds > thresh_segment
  print(x_test_real.shape)
  for i, y_pred in enumerate(y_preds):
    remove_small_objects(y_pred, thresh_obj, in_place=True)
    remove_small_holes(y_pred, thresh_hole, in_place=True)
    y_pred = y_pred*1.0
    y_axis, x_axis, _ = np.where(y_pred == 1.0)
    y_min = np.min(y_axis)
    y_min = y_min - border_pixel if y_min >= border_pixel else 0
    y_max = np.max(y_axis)
    y_max = y_max + border_pixel if y_max < h - border_pixel else h-1
    x_min = np.min(x_axis)
    x_min = x_min - border_pixel if x_min >= border_pixel else 0
    x_max = np.max(x_axis)
    x_max = x_max + border_pixel if x_max < w - border_pixel else w-1
    img_new = x_test_real[i][y_min:y_max, x_min:x_max]
    img_new = cv2.resize(img_new, (w, h))
    x_test_real[i] = img_new
  #print(x_test_real.shape)
  #cv2.imshow("g",x_test_real[0])
  #cv2.waitKey(0)
  y_preds = classify_model.predict(x_test_real)
  print(y_preds)
  if y_preds > thresh_classify:
    result_mess = f"COVID19: Positive\n {0:.2f}%".format(y_preds[0][0]*100)
  else:
    result_mess = "COVID19: Negative\n {0:.2f}%".format((1-y_preds[0][0])*100)
  return y_preds > thresh_classify
def browsecsv():
    global img_arr
    filename = filedialog.askopenfile(parent=root,mode='rb',title='Choose a file')
    h = filename.read()
    byte =  bytearray(h)
    image = Image.open(io.BytesIO(byte))
    img_arr = np.array(image)
    w, h = image.size
    canvas = Canvas(root, width = w, height = h) 
    canvas.pack()  
    photo = ImageTk.PhotoImage(image)
    root.photo = photo
    canvas.create_image(0, 0, anchor=NW, image=photo) 

def run():
        result = full_process(img_arr)
        print(result)
        messagebox.showinfo("resut",result_mess)
if __name__ == "__main__":
    init()
    #im =cv2.imread(r"F:\Nam_4\covid19\github_ucsd\COVID-CT-master\data_covid19\CT_NonCOVID\6%3.jpg")

    #result = full_process(im)
    #print(result)
    root = Tk()
    root.title("For people who can't afford Real-time PCR")
    root.geometry("500x400")
    bbutton = Button(root, text="Browse", command=browsecsv)
    bbutton.pack(pady = 10)
    abutton = Button(root, text="Run", command=run)
    abutton.pack(pady = 10)
    cbutton= Button(root, text="exit", command=root.destroy)
    cbutton.pack(pady = 10)
    root.mainloop() 