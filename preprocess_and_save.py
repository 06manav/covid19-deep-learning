import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import save
from numpy import load
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from tensorflow import keras
from tensorflow.image import resize
from tensorflow.image import ResizeMethod

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import skimage.io
import skimage.viewer
from skimage.color import rgb2gray
from skimage.util import img_as_float

def saveNumpyArrays(requiredShape, channelsNum, saving_path):
    covid_dataset_folder_path = 'COVID-19 Radiography Database/'
    class_names = ['COVID-19','NORMAL','Viral Pneumonia']
    numberOfImages = 219
    interpolation = 'bicubic'
    
    #requiredShape = 224
    if channelsNum == 1:
        datasetX = np.empty((3,numberOfImages,requiredShape,requiredShape), dtype=np.float64)
    else:
        datasetX = np.empty((3,numberOfImages,requiredShape,requiredShape,3), dtype=np.float64)
    datasetY = np.empty((3,numberOfImages), dtype=np.uint8)
    for classI in range(3):
        for imgI in range(numberOfImages):
            print(imgI)
            try:
                imgName = covid_dataset_folder_path + class_names[classI] +'/' + class_names[classI] \
                                                     + ' ('+ str(imgI+1) + ').png'                                                    
                image = skimage.io.imread(fname=imgName)
            except:
                imgName = covid_dataset_folder_path + class_names[classI] +'/' + class_names[classI] \
                                                 + '('+ str(imgI+1) + ').png'                                                    
                image = skimage.io.imread(fname=imgName)
            
            if channelsNum == 1 and len(image.shape) != 2:
                image = np.mean(image,2)
            if channelsNum == 3 and len(image.shape) == 2:
                image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            lx, ly = image.shape[:-1]
            cropF = 20
            #image = image[lx // cropF: - lx // cropF, ly // cropF: - ly // cropF]    
            image = img_as_float(image)
              
            image = resize(image, (requiredShape,requiredShape), anti_aliasing=True)
            image = tf.image.per_image_standardization(image)
            datasetX[classI,imgI] = image
            datasetY[classI,imgI] = classI
         
    print(datasetX.shape)
    print(datasetY.shape)
    
    save(str(saving_path) + 'datasetX.npy', datasetX)
    save(str(saving_path) + 'datasetY.npy', datasetY)
    

#change this to save images 
saving_path = ''
#this function saves images 
#input:   (required_image_shape, number of channels, saving path)
saveNumpyArrays(299, 3, saving_path)
loading_path = ''
datasetX = load(str(loading_path) + 'datasetX.npy')
datasetY = load(str(loading_path) + 'datasetY.npy')


for i in range(219):
    plt.subplot(1,2,1)
    plt.imshow(datasetX[0][i])
    plt.xlabel(datasetY[0][i])
    plt.subplot(1,2,2)
    img = datasetX[0][i]
    img = tf.image.per_image_standardization(img)
    img = (img) / 5 + 0.5
    print(np.std(img), np.mean(img))
    plt.imshow(img)
    plt.show()
    

