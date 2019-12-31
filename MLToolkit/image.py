# -*- coding: utf-8 -*-
# MLToolkit (mltoolkit)

__name__="mltk"

"""
MLToolkit - a verstile helping library for machine learning
===========================================================
'MLToolkit' is a Python package providing a set of user-friendly functions to 
help building machine learning models in data science research or production 
focused projects. It is compatible with and interoperate with popular data 
analysis, manipulation and machine learning libraries Pandas, Sci-kit Learn, 
Tensorflow, Statmodels, Catboost, XGboost, etc.

Main Features
-------------
- Data Extraction (SQL, Flatfiles, Images, etc.)
- Exploratory data analysis (statistical summary, univariate analysis, etc.)
- Feature Extraction and Engineering
- Model performance analysis, Explain Predictions and comparison between models
- Cross Validation and Hyper parameter tuning
- JSON input script for executing model building and scoring tasks.
- Model Building UI
- Auto ML (automated machine learning)
- Model Deploymet and Serving via RESTful  API

Author
------
- Sumudu Tennakoon

Links
-----
Website: http://sumudu.tennakoon.net/projects/MLToolkit
Github: https://mltoolkit.github.io/MLToolKit

License
-------
Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""

import pandas as pd
import numpy as np
import sys
import os
import re
import matplotlib.pyplot as plt 
from datetime import date

import PIL
import PIL.ImageEnhance
import PIL.ImageOps
import PIL.ImageFilter
import PIL.ImageDraw

def read_image_file(file_path, 
                    return_array=False, 
                    remove_transparancy=True, 
                    convert_grey_scale=True,
                    size=(28,28),
                    return_file_size=False,
                    show_image=False):
    """
    Parameters
    ----------
    file_path : str
    show_image : bool, default False
    return_size : bool, default False
    
    Returns
    -------
    image : PIL.Image
    
    """
        
    image = PIL.Image.open(file_path)
    
    if remove_transparancy:
        width, height = image.size
        if image.mode in ('RGBA', 'LA'):
            image_format = image.format
            canvas = PIL.Image.new('RGB', (width, height), (255, 255, 255))
            canvas.paste(image, (0, 0), image)
            image = canvas      
            image.format = image_format 
            
    if convert_grey_scale:
        if image.mode != 'L':
            image = image.convert('L')  #Convert to greyscale  
    
    image = image.resize(size=size, resample=PIL.Image.BOX)
        
    if show_image:
        image.show()
        
    image.load() # Close the file but keep the image data
    
    image = PIL.ImageOps.autocontrast(image)
                              
    if return_array:
        image = np.array(image) 
    
    if return_file_size:
        try:
            file_size = os.path.getsize(file_path)/1024 #Convert to KB
        except:
            file_size = None
        return image, file_size
    else:
        return image

def image_to_dataframe(file_path, image_column='Image', label_column='Label', size=(28,28), show_image=False):
    """
    Parameters
    ----------
    file_path : str
    show_image : bool, default False
    
    Returns
    -------
    image_dataframe : pandas.DataFrame
    """
    
    image = read_image_file(file_path, return_array=True, size=size, show_image=show_image)
    ImageDataFrame = pd.DataFrame(data=[[os.path.dirname(file_path), os.path.basename(file_path), image, None]], 
                                  columns=['Path', 'FileName', image_column, label_column])
    return ImageDataFrame
    
def read_image_folder(file_folder_path, show_image=False, size=(28,28)):
    """
    Parameters
    ----------
    file_folder_path : str
    show_image : bool, default False
    
    Returns
    -------
    ImagesDataFrame : pandas.DataFrame
    """
    files_list = os.listdir(file_folder_path)
    
    ImagesDataFrame = pd.DataFrame()
    
    for file_name in files_list:
        file_path = os.path.join(file_folder_path, file_name)    
        ImagesDataFrame = ImagesDataFrame.append(image_to_dataframe(file_path, size=size, show_image=False), ignore_index=True)
        
    return ImagesDataFrame


def prepare_image_dataset_to_model(ImagesDataFrame, image_column='Image', processed_image_column='ImageToModel', label_column='Label', 
                                   image_data_format='channels_last', size=(28,28)):
    """
    Parameters
    ----------
    ImagesDataFrame : pandas.DataFrame
    image_column : str
    label_column : str
    image_data_format : {'channels_first', 'channels_last'}
    size : (int, int), default (28,28)
    
    Returns
    -------
    x_data : np.array
    y_data : optional
    input_shape : tuple
    """
    
    img_rows = size[0]
    img_cols = size[1]
    n_images = ImagesDataFrame.shape[0]
    
    if image_data_format == 'channels_last':
        ImagesDataFrame[processed_image_column] = ImagesDataFrame[image_column].apply(np.reshape, args=((img_rows, img_cols, 1),))
        input_shape = (img_rows, img_cols, 1)
    elif image_data_format == 'channels_first':
        ImagesDataFrame[processed_image_column] = ImagesDataFrame[image_column].apply(np.reshape, args=((1, img_rows, img_cols),))
        input_shape = (1, img_rows, img_cols) 
    else:
        print('image_data_format not supported !')
        return ImagesDataFrame, (img_rows, img_cols)
    
    ImagesDataFrame[processed_image_column] = ImagesDataFrame[processed_image_column]/255.0
    #x_data = np.asarray(list(ImagesDataFrame[image_column].values), dtype ='float32') #convert interger image tensor to float
    #y_data = ImagesDataFrame[label_column].values
    
    '''
    if image_data_format == 'channels_last':
        x_data = x_data.reshape(n_images, img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    elif image_data_format == 'channels_first':
        x_data = x_data.reshape(n_images, 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)     
    else:
        print('image_data_format not supported !')
        return ImagesDataFrame, (img_rows, img_cols)
    '''
    #x_data = x_data/255.0 # Normalize grayscale to a number between 0 and 1
    
    #ImagesDataFrame[processed_image_column] = x_data
    #return x_data, y_data, input_shape
    
    return ImagesDataFrame, input_shape

def preview_image_dataset_to_model(ImagesDataFrame, image_column='Image', label_column='Label', classes=None, n_samples=1):
    """
    Parameters
    ----------
    ImagesDataFrame : pandas.DataFrame
    image_column : str
    label_column : str    
    n_samples : int, default 1
    
    Returns
    -------    
    image_plot : matplotlib.figure.Figure 
    """
    
    plt_columns = 3
    
    if n_samples > plt_columns:
        plt_rows = n_samples//plt_columns
        if n_samples-plt_rows*plt_columns > 0:
            plt_rows = plt_rows + 1
    else:
        plt_columns = n_samples
        plt_rows = 1
    
    samples = ImagesDataFrame.sample(n_samples)
    
    plt.figure(figsize=(plt_rows*10, plt_columns*5))
    
    i = 1
    for index, row  in samples.iterrows():
        plt.subplot(plt_rows, plt_columns, i)
        #plt.xticks([])
        #plt.yticks([])
        plt.grid(False)
        image_plot = plt.imshow(row[image_column])
        plt.xlabel('{} [{}]'.format(classes[row[label_column]], row[label_column]))
        i = i +1
        
    return image_plot

def encode_labels(y_data, n_classes):
    """
    Parameters
    ----------
    y_data : np.array(int)
    classes : dict
    
    Returns
    -------    
    y_data_enc : np.array
    """
    
    # convert class vectors to binary class matrices (One-hot encoding)
    y_data_enc = keras.utils.to_categorical(y_data, n_classes) 
    return y_data_enc

