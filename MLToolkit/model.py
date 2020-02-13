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

from datetime import datetime
import gc
import traceback
import gc
import os
import copy
from timeit import default_timer as timer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import warnings
warnings.filterwarnings("ignore")

from mltk.explore import *
from mltk.matrics import *
from mltk.deploy import *
from mltk.etl import *

def train_validate_test_split(DataFrame, ratios=(0.6,0.2,0.2)):
    """    
    Parameters
    ----------
    DataFrame : pandas.DataFrame
        DataFrame
    ratios : tuple
        E.g. 
            (train, validate, test) = (0.6, 0.25, 0.15)  
            (train, test) = (0.6, 0.4) ->  validate = test
            
    Returns
    -------
    TrainDataset : pandas.DataFrame
    ValidateDataset : pandas.DataFrame
    TestDataset : pandas.DataFrame
    """       
    from sklearn.model_selection import train_test_split
    N = len(DataFrame.index)
    
    if len(ratios)==3:        
        train_size = ratios[0]/np.sum(ratios)  
        test_size = ratios[2]/np.sum(ratios[1:3])        
        TrainDataset, TestDataset = train_test_split(DataFrame, train_size=train_size, random_state=42)
        ValidateDataset, TestDataset = train_test_split(TestDataset, test_size=test_size, random_state=42)        
    elif len(ratios)==2:
        train_size = ratios[0]/np.sum(ratios)
        TrainDataset, TestDataset = train_test_split(DataFrame, train_size=train_size, random_state=42)
        ValidateDataset = TestDataset
        print('Validate = Test')
    else:
        print('ERROR in splitting train, validate, test')
        return None, None, None
        
    n_train = len(TrainDataset.index)
    n_validate = len(ValidateDataset.index)
    n_test = len(TestDataset.index)
    
    print('Train Samples: {} [{:.1f}%]'.format(n_train, n_train/N*100))
    print('Validate Samples: {} [{:.1f}%]'.format(n_validate, n_validate/N*100))
    print('Test Samples: {} [{:.1f}%]'.format(n_test, n_test/N*100))        
    
    return TrainDataset, ValidateDataset, TestDataset

class MLModel():
    def __init__(self, model_attributes, model_parameters, sample_attributes, model_variables, variable_setup, target_variable, score_parameters, model_interpretation, model_evaluation, model_object=None):
        self.model_attributes = model_attributes
        self.model_parameters = model_parameters
        self.sample_attributes=sample_attributes
        self.model_variables = model_variables
        self.target_variable = target_variable
        self.score_parameters = score_parameters
        self.model_interpretation=model_interpretation
        self.model_evaluation = model_evaluation
        self.model_object = model_object 
        self.variable_setup = variable_setup
    
    def get_model_type(self):   
        try:
            return self.model_attributes['ModelType'] #regression/classification/clautering
        except:
            return None
    
    def set_model_object(self, model_object):
        self.model_object = model_object     
    
    def get_model_object(self):
        return self.model_object
    
    def get_model_algorithm(self):
        return self.model_parameters['MLAlgorithm']
        
    def get_identifier_variables(self):
        return self.sample_attributes['RecordIdentifiers']
        
    def get_target_variable(self):
        return self.target_variable
    
    def get_model_variables(self):
        return self.model_variables
    
    def get_variable_setup(self):
        return self.variable_setup

    def set_variable_setup(self, variable_setup):
        self.variable_setup = variable_setup
    
    def get_model_name(self):
        return self.model_attributes['ModelName']

    def get_model_id(self):
        return self.model_attributes['ModelID']
        
    def get_score_parameter(self, variable):
        return self.score_parameters[variable]

    def get_score_variable(self):
        return self.score_parameters['ScoreVariable']
		
    def get_score_label(self):
        return self.score_parameters['ScoreLabel']

    def get_score_edges(self):
        return self.score_parameters['Edges']
    
    def get_predicted_label(self):
        return self.score_parameters['PredictedLabel']

    def get_predict_threshold(self):
        return self.score_parameters['Threshold']
        
    def get_robustness_table(self):
        return self.model_evaluation['RobustnessTable']

    def get_roc_curve(self):
        return self.model_evaluation['ROCCurve']

    def get_precision_recall_curve(self):
        return self.model_evaluation['PrecisionRecallCurve']
    
    def get_auc(self, curve='roc'):
        try:
            if curve=='roc':
                return self.model_evaluation['ROC_AUC']
            if curve=='prc':
                return self.model_evaluation['PRC_AUC']
        except:
            return None
        
    def get_rmse(self):
        try:
            return self.model_evaluation['RMSE']
        except:
            return None

    def get_r2(self):
        try:
            return self.model_evaluation['R2']
        except:
            return None
        
    def set_score_edges(self, edges):
        self.score_parameters['Edges']=edges

    def set_predict_threshold(self, threshold):
        self.score_parameters['Threshold'] = threshold
    
    def get_model_data_stats(self):
        return self.sample_attributes['ModelDataStats']
    
    def get_model_manifest(self, save=False, save_path=''):       
        import json
        model_manifest= {'model_attributes':self.model_attributes,
                        'model_parameters':self.model_parameters,
                        'score_parameters':self.score_parameters,
                        'sample_attributes':self.sample_attributes,  
                        'roc_auc':self.get_auc(curve='roc'),
                        'prc_auc':self.get_auc(curve='prc'),
                        'rmse':self.get_rmse(),
                        'r2':self.get_r2(),
                        'robustness_table':self.get_robustness_table().to_dict(orient='dict')
                        }        
        if save==True:
            json_object = json.dumps(model_manifest)          
            f = open(os.path.join(save_path, '{}_ModelManifest.json'.format(self.get_model_id())),'w')
            f.write(json_object)
            f.close() 

        return model_manifest
    
    def plot_eval_matrics(self, figure=0, comparison=False, fig_size=(8, 6), layout=(2,3), dpi=80):
        import matplotlib.pyplot as plt
        from matplotlib import colors #For custom color maps
        
        description = self.get_model_id()
        model_type = self.get_model_type()

        if model_type=='classification':
            response_quantity_label = 'ResponseCount'
            bucket_mean_label = 'BucketPrecision'
            cum_bucket_mean_label = 'CumulativePrecision'
        elif model_type=='regression':
            response_quantity_label = 'ResponseAmount'
            bucket_mean_label = 'BucketMeanAmount'
            cum_bucket_mean_label = 'CumulativeMeanAmount'
        else:
            print('No Model Type provided')
            return None
        
        plt.figure(figure, figsize=fig_size, dpi=dpi)

        ax3 = plt.subplot(233)
        RobustnessTable = self.get_robustness_table().copy()[:-1]
        #plt.plot(thresholds, precision[:-1], 'o-', label='{} precision'.format(description))
        x_column = 'CumulativeBucketFraction'
        y_column = 'CumulativeResponseFraction'
        plt.plot(list(RobustnessTable[x_column])+[0.0], list(RobustnessTable[y_column])+[0.0], marker=".", linestyle='-', label='{}'.format(description)) #, marker='o'
        plt.legend()
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title('Bucket Fraction vs. Threshold Curve')
        
        max_x = max(RobustnessTable[x_column])*1.05
        max_y = max(RobustnessTable[y_column])*1.05   

        plt.plot([0, max(max_x, max_y)], [0, max(max_x, max_y)], color='black', linestyle='--')

        plt.ylim([0.0, max_y])
        plt.xlim([0.0, max_x])
            
        ax5 = plt.subplot(236)
        x_column = 'Mean{}'.format(self.get_score_parameter('ScoreVariable'))
        y_column = bucket_mean_label
        size_column = 'ResponseFraction' 

        max_x = max(RobustnessTable[x_column])*1.05
        max_y = max(RobustnessTable[y_column])*1.05  
        
        plt.plot([0, max(max_x, max_y, 1)], [0, max(max_x, max_y, 1)], color='black', linestyle='--')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title('Model Charateristics \n ResponseFraction ~ Marker size')
        
        plt.ylim([0.0, max_y])
        plt.xlim([0.0, max_x])
        
        size_scale=2000 
        size_offset=1      		
        
        if comparison:
            plt.scatter(RobustnessTable[x_column], RobustnessTable[y_column], s=size_offset+RobustnessTable[size_column].values*size_scale, label='{}'.format(description))
            plt.legend()
        else:
            color_column = 'BucketFraction'
            color_scale=100
            #plt.plot(thresholds, precision[:-1], 'o-', label='{} precision'.format(description))
            bounds = np.array([0.0, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 25.0, 50.0])
            norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
            RobustnessTable.sort_values(by=size_column, ascending=False, inplace=True)
            plt.scatter(RobustnessTable[x_column], RobustnessTable[y_column], c=RobustnessTable[color_column].values*color_scale, s=size_offset+RobustnessTable[size_column].values*size_scale, cmap='nipy_spectral', norm=norm, marker='s')
            cbar = plt.colorbar()
            cbar.set_label(color_column)  
        
        #############################
        if model_type=='regression':
            pass
        elif model_type=='classification':    
            roc_auc = self.get_auc(curve='roc')
            prc_auc = self.get_auc(curve='prc')
            
            plot_tile = {
                    'ROC':231,
                    'PvTH':232,
                    'RFvCF':233,
                    'PRC':234,               
                    'RvTH':235,
                    'PREDvACT': 236
                    }
            
            ax0 = plt.subplot(231) 
            ROCCurve = self.get_roc_curve()
            plt.plot(ROCCurve.FPR.values, ROCCurve.TPR.values, linestyle='-', label='{} (area = {:.2f}) '.format(description, roc_auc))
            plt.plot([0, 1], [0, 1], color='black', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC')
            plt.legend(loc="lower right")
            #plt.show()
            
            ax1 = plt.subplot(234) 
            PrecisionRecallCurve = self.get_precision_recall_curve()
            plt.plot(PrecisionRecallCurve['Recall'].values, PrecisionRecallCurve['Precision'].values, linestyle='-', label='{} (auc = {:.2f}) '.format(description, prc_auc))
            plt.legend()
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            
            ax2 = plt.subplot(232, sharey=ax1, sharex=ax3)
            plt.plot(PrecisionRecallCurve['Threshold'].values, PrecisionRecallCurve['Precision'].values, linestyle='-', label='{}'.format(description))
            #plt.plot(thresholds, recall[:-1], 'o-', label='{} recall'.format(description))
            plt.legend()
            plt.xlabel('Threshold')
            plt.ylabel('Precision')
            plt.title('Precision vs. Threshold Curve')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            
            ax4 = plt.subplot(235, sharex=ax2)
            #plt.plot(thresholds, precision[:-1], 'o-', label='{} precision'.format(description))
            plt.plot(PrecisionRecallCurve['Threshold'].values, PrecisionRecallCurve['Recall'].values, linestyle='-', label='{}'.format(description))
            plt.legend()
            plt.xlabel('Threshold')
            plt.ylabel('Recall')
            plt.title('Recall vs. Threshold Curve')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
        #######################
        
        plt.subplots_adjust(hspace=0.4)
        #plt.show()

def load_tensorflow_model(file_path):
    from tensorflow import keras 
    model = keras.models.load_model(file_path)
    return model
        
def load_object(file_name):
    try:
        import pickle
        pickle_in = open(file_name,'rb')
        print('Loading model from file {}'.format(file_name))
        object = pickle.load(pickle_in)
        pickle_in.close()
        return object
    except:
        print('ERROR in loading model: {}'.format(traceback.format_exc()))
        return None
    
def save_object(object_to_save, file_name):
    try:
        import pickle
        pickle_out = open(file_name,'wb')
        print('Saving model to file {}'.format(file_name))
        pickle.dump(object_to_save, pickle_out)
        pickle_out.close()
    except:
        print('ERROR in saving model: {}'.format(traceback.format_exc()))

def save_model(Model, file_path):
    """
    Parameters
    ----------
    Model : mltk.MLModel
        MLModel Object
    file_path : str, dafault ''
        Path to file including the file name.    
    
    Returns
    -------
    None
    """
    if Model.model_parameters['MLAlgorithm']=='NN':
        Model.model_object.save(os.path.splitext(file_path)[0]+'.tfh5')
        Model.set_model_object(None)       
    save_object(Model, file_path)

def load_model(file_path):
    """
    Parameters
    ----------
    file_path : str, dafault ''
        Path to file including the file name.    
    
    Returns
    -------
    Model : mltk.MLModel
        MLModel Object
    """
    Model = load_object(file_path)
    if Model.model_parameters['MLAlgorithm']=='NN':
        model = load_tensorflow_model(os.path.splitext(file_path)[0]+'.tfh5')
        Model.set_model_object(model)
    return Model

def build_regression_model(x_train, y_train, model_variables, target_variable=None, model_parameters=None):
    """
    Parameters
    ----------
    x_train : np.array
    y_train : np.array
    model_variables : list(str)
    target_variable : str
    model_parameters : dict
    
    Returns
    -------
    model : mltk.MLModel
    output : dict
    model_fit_time : float
    """
    from timeit import default_timer as timer
    
    MLAlgorithm = model_parameters['MLAlgorithm']    
    
    if MLAlgorithm == 'LREG':
        import statsmodels as sm
        import statsmodels.discrete.discrete_model as sm
        import statsmodels.api as smapi

        maxiter=model_parameters['MaxIterations']    
        model=smapi.OLS(y_train, x_train)

        start_time = timer()  
        model=model.fit(maxiter=maxiter)
        model_fit_time = timer() - start_time

        output = model.summary(xname=list(model_variables))
    
    elif MLAlgorithm == 'RFREG':
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.datasets import make_regression

        NTrees = model_parameters['NTrees']
        MaxDepth = model_parameters['MaxDepth']
        Processors = model_parameters['Processors']
        MinSamplesToSplit = model_parameters['MinSamplesToSplit']
    
        try:
            verbose = int(model_variables['Verbose'])
        except:
            verbose = 0
               
        model = RandomForestRegressor(max_depth=MaxDepth, n_jobs=Processors, verbose=verbose, min_samples_split=MinSamplesToSplit, n_estimators=NTrees)
        
        start_time = timer()      
        model.fit(x_train, y_train)  
        model_fit_time = timer() - start_time
    
        Importances = model.feature_importances_
        stdev = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)    
        output = pd.DataFrame({'Features':model_variables, 'Importances':Importances, 'stdev':stdev}).sort_values(by='Importances', ascending=False)       

    return model, output, model_fit_time
	
def build_logit_model(x_train, y_train, model_variables, target_variable, model_parameters):
    """
    Parameters
    ----------
    x_train : np.array
    y_train : np.array
    model_variables : list(str)
    target_variable : str
    model_parameters : dict
    
    Returns
    -------
    model : mltk.MLModel
    output : dict
    model_fit_time : float
    """
    
    import statsmodels as sm
    import statsmodels.discrete.discrete_model as sm
    import statsmodels.api as smapi
    
    maxiter=model_parameters['MaxIterations']    
    model=smapi.Logit(y_train, x_train)
    
    startTime = timer()  
    model=model.fit(maxiter=maxiter)
    model_fit_time = timer() - startTime
    
    output = model.summary(xname=list(model_variables))
        
    return model, output, model_fit_time

###############################################################################
def stack_nn_layers_tf(architecture, print_summary=False):
    """
    Parameters
    ----------
    architecture : dict
        list of layers
    print_summary : bool, default False
        prints model summary if True
        
    Returns
    -------
    model : keras.engine.sequential.Sequential
        Keras Sequential model object
    """
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, MaxPooling1D, Flatten
    from keras import regularizers

    def set_default_param(config, param, default=None):    
        if param not in config:
            config.update({param:default})
        else:
            if param in ('kernel_regularizer', 'bias_regularizer', 'activity_regularizer'):
                l1 = config[param]['l1'] if ('l1' in config) else 0.0
                l2 = config[param]['l2'] if ('l2' in config) else 0.0 
                config[param] = regularizers.l1_l2(l1=l1, l2=l2)
        return config
    
    model = Sequential()
    
    for layer in architecture['layers']:
        try:
            layer['class_name'] = layer['type']
        except:
            pass
        try:
            layer['config']['name'] = layer['name']
        except:
            pass        
        config = layer['config']
        config = set_default_param(config, param='name', default=None)
        config = set_default_param(config, param='strides', default=None)
        config = set_default_param(config, param='padding', default='valid') 
        config = set_default_param(config, param='data_format', default=None) 
        config = set_default_param(config, param='kernel_regularizer', default=None) 
        config = set_default_param(config, param='bias_regularizer', default=None) 
        config = set_default_param(config, param='activity_regularizer', default=None)
        
        if layer['class_name'] in ('Conv1D', 'Conv2D', 'Conv3D') and config['strides']==None:
            config['strides'] = 1 #Set to default value
        
        if layer['class_name']=='Dense':
            if layer['position']=='input':
                model.add(Dense(name=config['name'], units=config['units'], activation=config['activation'], input_shape=config['input_shape'], kernel_regularizer=config['kernel_regularizer'], activity_regularizer=config['activity_regularizer'], bias_regularizer=config['bias_regularizer']))
            else:
                model.add(Dense(name=config['name'], units=config['units'], activation=config['activation'], kernel_regularizer=config['kernel_regularizer'], activity_regularizer=config['activity_regularizer'], bias_regularizer=config['bias_regularizer']))
        elif layer['class_name']=='Dropout': 
             model.add(Dropout(name=config['name'], rate=config['rate']))   
        elif layer['class_name']=='Conv2D':
            if layer['position']=='input':
                model.add(Conv2D(name=config['name'], filters=config['filters'], kernel_size=config['kernel_size'], strides=config['strides'], padding=config['padding'], data_format=config['data_format'], activation=config['activation'], input_shape=config['input_shape'],  kernel_regularizer=config['kernel_regularizer'], activity_regularizer=config['activity_regularizer'], bias_regularizer=config['bias_regularizer'])) 
            elif layer['position']=='hidden':
                model.add(Conv2D(name=config['name'], filters=config['filters'], kernel_size=config['kernel_size'], strides=config['strides'], padding=config['padding'], data_format=config['data_format'], activation=config['activation'], kernel_regularizer=config['kernel_regularizer'], activity_regularizer=config['activity_regularizer'], bias_regularizer=config['bias_regularizer']))
        elif layer['class_name']=='MaxPooling2D':
                model.add(MaxPooling2D(name=config['name'], pool_size=config['pool_size'], strides=config['strides'], padding=config['padding'], data_format=config['data_format']))
        elif layer['class_name']=='Conv1D':
            if layer['position']=='input':
                model.add(Conv1D(name=config['name'], filters=config['filters'], kernel_size=config['kernel_size'], strides=config['strides'], padding=config['padding'], data_format=config['data_format'], activation=config['activation'], input_shape=config['input_shape'], kernel_regularizer=config['kernel_regularizer'], activity_regularizer=config['activity_regularizer'], bias_regularizer=config['bias_regularizer'])) 
            else:
                model.add(Conv1D(name=config['name'], filters=config['filters'], kernel_size=config['kernel_size'], strides=config['strides'], padding=config['padding'], data_format=config['data_format'], activation=config['activation'], kernel_regularizer=config['kernel_regularizer'], activity_regularizer=config['activity_regularizer'], bias_regularizer=config['bias_regularizer']))
        elif layer['class_name']=='MaxPooling1D':
                model.add(MaxPooling1D(name=config['name'], pool_size=config['pool_size'], strides=config['strides'], padding=config['padding'], data_format=config['data_format']))
        elif layer['class_name']=='Conv3D':
            if layer['position']=='input':
                model.add(Conv3D(name=config['name'], filters=config['filters'], kernel_size=config['kernel_size'], strides=config['strides'], padding=config['padding'], data_format=config['data_format'], activation=config['activation'], input_shape=config['input_shape'], kernel_regularizer=config['kernel_regularizer'], activity_regularizer=config['activity_regularizer'], bias_regularizer=config['bias_regularizer'])) 
            else:
                model.add(Conv3D(name=config['name'], filters=config['filters'], kernel_size=config['kernel_size'], strides=config['strides'], padding=config['padding'], data_format=config['data_format'], activation=config['activation'], kernel_regularizer=config['kernel_regularizer'], activity_regularizer=config['activity_regularizer'], bias_regularizer=config['bias_regularizer']))
        elif layer['class_name']=='MaxPooling3D':
                model.add(MaxPooling3D(name=config['name'], pool_size=config['pool_size'], strides=config['strides'], padding=config['padding'], data_format=config['data_format']))
        elif layer['class_name']=='Flatten':
                model.add(Flatten(name=config['name'], data_format=config['data_format']))
    
    if print_summary:
        print(model.summary())
                
    return model

def stack_nn_layers_tf_depreciate_in_0_2_0(architecture, print_summary=False):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, MaxPooling1D, Flatten
    
    model = Sequential()
    
    for key, layer in architecture.items():
        if layer['type']=='Dense':
            if layer['position']=='input':
                model.add(Dense(units=layer['units'], activation=layer['activation'], input_shape=layer['input_shape']))
            elif layer['position']=='hidden':
                model.add(Dense(units=layer['units'], activation=layer['activation']))
            elif layer['position']=='output':
                model.add(Dense(units=layer['units'], activation=layer['activation'])) #, output_shape=layer['output_shape']
        elif layer['type']=='Dropout': 
             model.add(Dropout(rate=layer['rate']))   
        elif layer['type']=='Conv2D':
            if layer['position']=='input':
                model.add(Conv2D(filters=layer['filters'], kernel_size=layer['kernel_size'], strides=layer['strides'], padding=layer['padding'], data_format='channels_last', activation=layer['activation'], input_shape=layer['input_shape'])) #, strides=(2, 2)
            elif layer['position']=='hidden':
                model.add(Conv2D(filters=layer['filters'], kernel_size=layer['kernel_size'], strides=layer['strides'], padding=layer['padding'], data_format='channels_last', activation=layer['activation'])) #, strides=(2, 2)
        elif layer['type']=='MaxPooling2D':
                model.add(MaxPooling2D(pool_size=layer['pool_size'], strides=None, padding=layer['padding'], data_format='channels_last'))
        elif layer['type']=='MaxPooling1D':
                model.add(MaxPooling1D(pool_size=layer['pool_size'], strides=None, padding=layer['padding'], data_format='channels_last'))
        elif layer['type']=='Flatten':
                model.add(Flatten())
    
    if print_summary:
        model.summary()
                
    return model

def stack_nn_layers(architecture, print_summary=False):
    architecture_ = copy.deepcopy(architecture)
    try:
        return stack_nn_layers_tf(architecture_, print_summary=print_summary)
    except:
        return stack_nn_layers_tf_depreciate_in_0_2_0(architecture_, print_summary=print_summary)
###############################################################################
    
def build_nn_model(x_train, y_train, x_validate, y_validate, model_variables, target_variable, model_parameters):
    """
    Parameters
    ----------
    x_train : np.array
    y_train : np.array
    x_validate : np.array
    y_validate : np.array  
    model_variables : list(str)
    target_variable : str
    model_parameters : dict
    
    Returns
    -------
    model : mltk.MLModel
    output : dict
    model_fit_time : float
    """
    
    import tensorflow as tf
    import tensorflow.keras as keras
    #from tensorflow.keras.models import Sequential
    #from tensorflow.keras.layers import Dense, Dropout

    #tf.enable_eager_execution()
    
    batch_size = model_parameters['BatchSize']
    #input_shape = model_parameters['InputShape']
    epochs = model_parameters['Epochs']   
    metrics = model_parameters['EvalMatrics']    
    num_classes = model_parameters['NumClasses']      
    architecture = model_parameters['Architecture']
    
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_validate = keras.utils.to_categorical(y_validate, num_classes)
    
    ###########################################################################    
    print('Input shape:', x_train.shape)
    model = stack_nn_layers(architecture)
    model.summary()  
    ###########################################################################
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(), #keras.optimizers.RMSprop(),
                  metrics=metrics) #['accuracy']

    # Create Tensorflow Session
    sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0, log_device_placement=True)) #device_count={'GPU': 0}, 
    keras.backend.set_session(sess)
    
    startTime = timer()
    hist = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_validate, y_validate))
    model_fit_time = timer() - startTime
    
    output = pd.DataFrame(data = hist.history)
    
    score = model.evaluate(x_validate, y_validate, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
       
    return model, output, model_fit_time
    
    
def build_rf_model(x_train, y_train, model_variables, target_variable, model_parameters):
    """
    Parameters
    ----------
    x_train : np.array
    y_train : np.array
    model_variables : list(str)
    target_variable : str
    model_parameters : dict
    
    Returns
    -------
    model : mltk.MLModel
    output : dict
    model_fit_time : float
    """
    from sklearn.ensemble import RandomForestClassifier
    
    try:
        verbose = int(model_variables['Verbose'])
    except:
        verbose = 0
        
    NTrees = model_parameters['NTrees']
    MaxDepth = model_parameters['MaxDepth']
    Processors = model_parameters['Processors']
    MinSamplesToSplit = model_parameters['MinSamplesToSplit']
    model = RandomForestClassifier(n_estimators=NTrees, max_depth=MaxDepth, n_jobs=Processors, verbose=verbose, min_samples_split=MinSamplesToSplit)

    startTime = timer()      
    model.fit(x_train, y_train)  
    model_fit_time = timer() - startTime
    
    Importances = model.feature_importances_
    stdev = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)    
    output = pd.DataFrame({'Features':model_variables, 'Importances':Importances, 'stdev':stdev}).sort_values(by='Importances', ascending=False)       
        
    return model, output, model_fit_time

def build_cbst_model(x_train, y_train, x_validate, y_validate, model_variables, target_variable, model_parameters):
    """
    Parameters
    ----------
    x_train : np.array
    y_train : np.array
    model_variables : list(str)
    target_variable : str
    model_parameters : dict
    
    Returns
    -------
    model : mltk.MLModel
    output : dict
    model_fit_time : float
    """
    
    import catboost
    from catboost import CatBoostClassifier, Pool
    import numpy as np
    
    # Model Params   
    num_trees = model_parameters['NTrees']
    depth = model_parameters['MaxDepth']
    learning_rate = model_parameters['LearningRate']   
    imbalanced = model_parameters['Imbalanced']
    loss_function = model_parameters['LossFunction']
    eval_metric = model_parameters['EvalMatrics'] 
    task_type = model_parameters['TaskType']
    thread_count = model_parameters['Processors'] 
    use_best_model = model_parameters['UseBestModel'] 
    verbose=int((num_trees+9)/10)

    if imbalanced==True:
        sample_size = len(y_train)
        actual_positives = np.sum(y_train)
        actual_negatives= sample_size-actual_positives
        scale_pos_weight = (1.0)*actual_negatives/actual_positives
    else:
        scale_pos_weight=1.0
    
    model = CatBoostClassifier(num_trees=num_trees,
                               depth=depth,                               
                               learning_rate=learning_rate,
                               loss_function=loss_function,
                               eval_metric=eval_metric,
                               scale_pos_weight=scale_pos_weight,
                               verbose=True, 
                               task_type = task_type,
                               thread_count=thread_count)
    
    startTime = timer()
    model.fit(x_train, y_train, eval_set=(x_validate, y_validate), use_best_model=use_best_model, verbose=verbose)
    model_fit_time = timer() - startTime

    Importances = model.feature_importances_
    #stdev = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)    
    output = pd.DataFrame({'Features':model_variables, 'Importances':Importances}).sort_values(by='Importances', ascending=False)        #, 'stdev':stdev
    
    score = model.score(x_validate, y_validate)
    print('Test accuracy:', score)
       
    return model, output, model_fit_time        

def build_xgbst_model(x_train, y_train, x_validate, y_validate, model_variables, target_variable, model_parameters):
    """
    Parameters
    ----------
    x_train : np.array
    y_train : np.array
    model_variables : list(str)
    target_variable : str
    model_parameters : dict
    
    Returns
    -------
    model : mltk.MLModel
    output : dict
    model_fit_time : float
    https://xgboost.readthedocs.io/en/latest/python/python_api.html
    """
    
    import xgboost
    from xgboost import XGBClassifier
    import numpy as np
    
    # Performance enhanced data structure
    #data_dmatrix = xgboost.DMatrix(data=x_train, label=y_train)

    # Model Params 
    num_trees = model_parameters['NTrees'] #n_estimators
    depth = model_parameters['MaxDepth'] #max_depth
    learning_rate = model_parameters['LearningRate'] 
    loss_function = model_parameters['LossFunction'] #objective
    eval_metric = model_parameters['EvalMatrics'] #eval_metric
    early_stopping_rounds = model_parameters['EarlyStopAttempts']
    samples_per_tree = model_parameters['SamplesRatioPerTree'] #subsample
    features_per_tree = model_parameters['FeaturesRatioPerTree'] #colsample_bytree
    #gamma = model_parameters['Gamma'] 
    l1 = model_parameters['Regularization']['L1'] #alpha (L1 regularization on leaf weights)
    l2 = model_parameters['Regularization']['L2'] #lambda (L2 regularization on leaf weights)
    verbose = model_parameters['Verbose'] 
    
    model = XGBClassifier(objective=loss_function, 
                          n_estimators=num_trees,
                          max_depth=depth,                               
                          learning_rate=learning_rate,
                          eval_metric=eval_metric,
                          verbose=verbose)    
    #imbalanced = model_parameters['Imbalanced']
    # 
    #task_type = model_parameters['TaskType']
    #thread_count = model_parameters['Processors'] 
    #use_best_model = model_parameters['UseBestModel'] 
    #verbose=int((num_trees+9)/10)

    #if imbalanced==True:
    #    sample_size = len(y_train)
    #    actual_positives = np.sum(y_train)
    #    actual_negatives= sample_size-actual_positives
    #    scale_pos_weight = (1.0)*actual_negatives/actual_positives
    #else:
    #    scale_pos_weight=1.0
    
    startTime = timer()
    model.fit(x_train, y_train, eval_set=(x_validate, y_validate), early_stopping_rounds=early_stopping_rounds)
    model_fit_time = timer() - startTime

    Importances = model.feature_importances_
    #stdev = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)    
    output = pd.DataFrame({'Features':model_variables, 'Importances':Importances}).sort_values(by='Importances', ascending=False)        #, 'stdev':stdev
    
    score = model.score(x_validate, y_validate)
    print('Test accuracy:', score)
       
    return model, output, model_fit_time

def build_lgbm_model(x_train, y_train, x_validate, y_validate, model_variables, target_variable, model_parameters):
    """
    Parameters
    ----------
    x_train : np.array
    y_train : np.array
    model_variables : list(str)
    target_variable : str
    model_parameters : dict
    
    Returns
    -------
    model : mltk.MLModel
    output : dict
    model_fit_time : float
    https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMModel.html#lightgbm.LGBMModel
    """
    
    import lightgbm 
    from lightgbm  import LGBMClassifier
    import numpy as np

    # Model Params 
    num_trees = model_parameters['NTrees'] #n_estimators
    depth = model_parameters['MaxDepth'] #max_depth
    learning_rate = model_parameters['LearningRate'] 
    loss_function = model_parameters['LossFunction'] #objective
    eval_metric = model_parameters['EvalMatrics'] #eval_metric
    early_stopping_rounds = model_parameters['EarlyStopAttempts']
    samples_per_tree = model_parameters['SamplesRatioPerTree'] #subsample
    features_per_tree = model_parameters['FeaturesRatioPerTree'] #colsample_bytree
    l1 = model_parameters['Regularization']['L1'] #alpha (L1 regularization on leaf weights)
    l2 = model_parameters['Regularization']['L2'] #lambda (L2 regularization on leaf weights)
    thread_count = model_parameters['Processors']  #n_jobs 
    verbose = model_parameters['Verbose'] 
    
    model = LGBMClassifier(objective=loss_function, 
                          n_estimators=num_trees,
                          max_depth=depth,                               
                          learning_rate=learning_rate,
                          eval_metric=eval_metric,
                          n_jobs=thread_count)    
    #imbalanced = model_parameters['Imbalanced']
    # 
    #task_type = model_parameters['TaskType']
    
    #use_best_model = model_parameters['UseBestModel'] 
    #verbose=int((num_trees+9)/10)

    #if imbalanced==True:
    #    sample_size = len(y_train)
    #    actual_positives = np.sum(y_train)
    #    actual_negatives= sample_size-actual_positives
    #    scale_pos_weight = (1.0)*actual_negatives/actual_positives
    #else:
    #    scale_pos_weight=1.0
    
    startTime = timer()
    model.fit(x_train, y_train, eval_set=(x_validate, y_validate), early_stopping_rounds=early_stopping_rounds, verbose=verbose)
    model_fit_time = timer() - startTime

    Importances = model.feature_importances_
    #stdev = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)    
    output = pd.DataFrame({'Features':model_variables, 'Importances':Importances}).sort_values(by='Importances', ascending=False)        #, 'stdev':stdev
    
    score = model.score(x_validate, y_validate)
    print('Test accuracy:', score)
       
    return model, output, model_fit_time
    
def model_dataset_to_array(Dataset, model_variables, target_variable, is_image_data=False):
    """
    TO DO: Find a more robust solution for this
    is_image_data : bool, defult False
        Converting Pandas series to numpy array will loose the shape when the data is mulidimentional (image data)
    """
    
    if is_image_data:
        x_data = np.asarray(list(Dataset[model_variables[0]].values), dtype ='float32') # Fails when model_varibles provided as a list
    else:
        x_data = Dataset[model_variables].values
    
    y_data = Dataset[target_variable].values
    y_data_act = y_data
    
    return x_data, y_data, y_data_act
    
    
    
def build_ml_model(TrainDataset, ValidateDataset, TestDataset, model_variables, target_variable, 
                 variable_setup, model_attributes, sample_attributes, model_parameters, score_parameters,
                   return_model_object=False, show_results=False, show_plot=False):
    """
    Parameters
    ----------
    TrainDataset : pandas.DataFrame
    ValidateDataset : pandas.DataFrame
    TestDataset : pandas.DataFrame
    model_variables : dict
    variable_setup : dict
    target_variable : str 
    model_attributes : dict
    sample_attributes : dict
    model_parameters : dict
    score_parameters : dict
    return_model_object : bool, default False
    show_results : bool, default False
    show_plot : bool, default False
    
    Returns
    -------
    Model : mltk.MLModel
        MLModel Object
    """
    ###############################################################################
    # Create deep copies of each dictionary
    model_variables = copy.deepcopy(model_variables)
    target_variable = copy.deepcopy(target_variable)
    model_attributes = copy.deepcopy(model_attributes)
    sample_attributes = copy.deepcopy(sample_attributes)
    model_parameters = copy.deepcopy(model_parameters)
    score_parameters = copy.deepcopy(score_parameters)
    ###############################################################################
    ml_algorithm = model_parameters['MLAlgorithm']
    model_type = model_attributes['ModelType']
    data_format = sample_attributes['DataFormat']
    
    if data_format=='image' or ml_algorithm=='CNN': # "ml_algorithm=='CNN'" WILL DEPRECIATE IN v0.2.0
        is_image_data = True
    else:
        is_image_data = False
    ###############################################################################
    if model_type=='classification':
        target_class = score_parameters['TargetClass']
        class_labels_map = sample_attributes['ClassLabelsMap'] 
        model_parameters['NumClasses'] = len(class_labels_map.keys())
    try:    
        verbose = model_parameters['Verbose']
    except:
        verbose = True
    ###############################################################################
    
    # TRAIN DATASET
    x_train, y_train, y_train_act = model_dataset_to_array(TrainDataset, 
                                                           model_variables, 
                                                           target_variable, 
                                                           is_image_data=is_image_data)
    
    print('Train samples: {} loded...'.format(x_train.shape[0]))
    ###############################################################################
    # VALIDATE DATASET
    x_validate, y_validate, y_validate_act = model_dataset_to_array(ValidateDataset, 
                                                                   model_variables, 
                                                                   target_variable, 
                                                                   is_image_data=is_image_data)
    
    print('Validate samples: {} loded...'.format(x_validate.shape[0]))
    ###############################################################################
    # TEST DATASET
    x_test, y_test, y_test_act = model_dataset_to_array(TestDataset, 
                                                           model_variables, 
                                                           target_variable, 
                                                           is_image_data=is_image_data)
    
    print('Test samples: {} loded...'.format(x_test.shape[0]))
    ###############################################################################
    
    model_interpretation={}

    model_evaluation={}
    
    # ModelAttributes 
    import mltk
    model_attributes['MLTKVersion'] = mltk.__version__
    model_attributes['BuiltTime'] = datetime.now().strftime('%Y%m%d%H%M%S')
    model_attributes['ModelID'] = model_attributes['ModelName'].replace(' ', '').upper()+model_parameters['MLAlgorithm'].replace(' ', '').upper()+model_attributes['BuiltTime']  
    model_attributes['ModelFitTime']=-1

    # SampleAttribues
    RecordIdentifiers = sample_attributes['RecordIdentifiers']
    sample_attributes['TrainSize'] = len(TrainDataset.index)
    sample_attributes['ValidateSize'] = len(ValidateDataset.index)
    sample_attributes['TestSize'] = len(TestDataset.index)
	#
    sample_attributes['TrainValidateTestRatio'] = '{}'.format(np.round(np.array([sample_attributes['TrainSize'],sample_attributes['ValidateSize'],sample_attributes['TestSize']])/(sample_attributes['TrainSize']+sample_attributes['ValidateSize']+sample_attributes['TestSize']),2))
    sample_attributes['TrainResponseRate'] = len(TrainDataset.loc[TrainDataset[target_variable]==1].index)/sample_attributes['TrainSize'] 
    sample_attributes['ValidateResponseRate'] = len(ValidateDataset.loc[ValidateDataset[target_variable]==1].index)/sample_attributes['ValidateSize']
    sample_attributes['TestResponseRate'] = len(TestDataset.loc[TestDataset[target_variable]==1].index)/sample_attributes['TestSize'] 

    # ScoreParameters
    edges = score_parameters['Edges']
    threshold=score_parameters['Threshold']
    quantiles = score_parameters['Quantiles']
    quantile_label = score_parameters['QuantileLabel']
    score_variable=score_parameters['ScoreVariable']
    score_label=score_parameters['ScoreLabel']
    predicted_label=score_parameters['PredictedLabel']
    
    ###############################################################################
    print(model_attributes)
    print(model_parameters)
    print(sample_attributes['SampleDescription'])
    print(score_parameters)
    ###############################################################################
    
    # Model fit routine selection
    if model_type == 'classification':
        if ml_algorithm=='LGR':
            ###############################################################################
            import statsmodels
            model_attributes['MLTool'] = 'statsmodels={}'.format(statsmodels.__version__) 
            # Logit Model
            model, summary, model_fit_time = build_logit_model(x_train, y_train, model_variables, target_variable, model_parameters)
            model_interpretation['ModelSummary'] = summary    

    #        y_pred_prob = model.predict(x_validate)
    #        ValidateDataset[score_variable]=y_pred_prob
            
            y_pred_prob = model.predict(x_test)
            TestDataset[score_variable]=y_pred_prob
            ###############################################################################    
        elif ml_algorithm=='RF':
            ###############################################################################
            import sklearn
            model_attributes['MLTool'] = 'sklearn={}'.format(sklearn.__version__) 
            # Random Forest Model
            model, summary, model_fit_time= build_rf_model(x_train, y_train, model_variables, target_variable, model_parameters)
            model_interpretation['ModelSummary'] = summary  
            
    #        y_pred_prob = model.predict(x_validate)
    #        ValidateDataset[score_variable]=y_pred_prob
            
            y_pred_prob = model.predict_proba(x_test).T #[:,1] 
            TestDataset[score_variable]=y_pred_prob[1]
            
            for i in range(len(y_pred_prob)): # ADDED TO SUPPORT MULTI-CLASS CLASSIFICATION
                print('{}_{}'.format(score_variable,i)) 
                TestDataset['{}_{}'.format(score_variable,i)] = y_pred_prob[i]
            ###############################################################################
            
        elif ml_algorithm in ['NN', 'CNN']:
            ###############################################################################
            import tensorflow
            import tensorflow.keras
            model_attributes['MLTool'] = 'tensorflow={}; keras={}'.format(tensorflow.__version__, tensorflow.keras.__version__) 
            # Deep Feed Forward Model using TensorFlow
            model, summary, model_fit_time= build_nn_model(x_train, y_train, x_validate, y_validate, model_variables, target_variable, model_parameters)
            model_interpretation['ModelSummary'] = summary  
            
    #        y_pred_prob = model.predict(x_validate)
    #        ValidateDataset[score_variable]=y_pred_prob
            
            y_pred_prob = model.predict(x_test, verbose=int(verbose), batch_size=model_parameters['BatchSize']).T #[:,1] 
            TestDataset[score_variable]=y_pred_prob[1]
            
            for i in range(len(y_pred_prob)): # ADDED TO SUPPORT MULTI-CLASS CLASSIFICATION
                #print('{}_{}'.format(score_variable,i)) 
                TestDataset['{}_{}'.format(score_variable,i)] = y_pred_prob[i]
                
            ###############################################################################   
        elif ml_algorithm=='CBST':
            ###############################################################################
            import catboost
            model_attributes['MLTool'] = 'catboost={}'.format(catboost.__version__)  
            # Cat Boost Model
            model, summary, model_fit_time= build_cbst_model(x_train, y_train, x_validate, y_validate, model_variables, target_variable, model_parameters)
            model_interpretation['ModelSummary'] = summary  
            
            y_pred_prob = model.predict_proba(x_test, verbose=verbose, thread_count=model_parameters['Processors']).T #[:,1]           
            TestDataset[score_variable]=y_pred_prob[1]
            
            for i in range(len(y_pred_prob)): # ADDED TO SUPPORT MULTI-CLASS CLASSIFICATION
                print('{}_{}'.format(score_variable,i)) 
                TestDataset['{}_{}'.format(score_variable,i)] = y_pred_prob[i]
    
            ###############################################################################
        elif ml_algorithm=='XGBST':
            ###############################################################################
            import xgboost
            model_attributes['MLTool'] = 'xgboost={}'.format(xgboost.__version__) 
            # Cat Boost Model
            model, summary, model_fit_time= build_xgbst_model(x_train, y_train, x_validate, y_validate, model_variables, target_variable, model_parameters)
            model_interpretation['ModelSummary'] = summary  
            
            y_pred_prob = model.predict_proba(x_test, verbose=True, thread_count=model_parameters['Processors']).T #[:,1] 
            TestDataset[score_variable]=y_pred_prob[1]
            
            for i in range(len(y_pred_prob)): # ADDED TO SUPPORT MULTI-CLASS CLASSIFICATION
                print('{}_{}'.format(score_variable,i)) 
                TestDataset['{}_{}'.format(score_variable,i)] = y_pred_prob[i]
                
            ###############################################################################
        elif ml_algorithm=='LGBM':
            ###############################################################################
            import lightgbm
            model_attributes['MLTool'] = 'lightgbm={}'.format(lightgbm.__version__) 
            # Cat Boost Model
            model, summary, model_fit_time= build_lgbm_model(x_train, y_train, x_validate, y_validate, model_variables, target_variable, model_parameters)
            model_interpretation['ModelSummary'] = summary  
            
            y_pred_prob = model.predict_proba(x_test, verbose=int(verbose), thread_count=model_parameters['Processors']).T #[:,1] 
            TestDataset[score_variable]=y_pred_prob[1]
            
            for i in range(len(y_pred_prob)): # ADDED TO SUPPORT MULTI-CLASS CLASSIFICATION
                print('{}_{}'.format(score_variable,i)) 
                TestDataset['{}_{}'.format(score_variable,i)] = y_pred_prob[i]
            ###############################################################################
        else:
            raise Exception('No Classification MLAlgorithm is specified in the model_parameters!')
        
    elif model_type == 'regression':
        if ml_algorithm=='LREG':
            ###############################################################################
            import statsmodels
            model_attributes['MLTool'] = 'statsmodels={}'.format(statsmodels.__version__) 
            # Logit Model
            model, summary, model_fit_time = build_regression_model(x_train, y_train, model_variables, target_variable, model_parameters)
            model_interpretation['ModelSummary'] = summary    

    #        y_pred_prob = model.predict(x_validate)
    #        ValidateDataset[score_variable]=y_pred_prob
            
            y_pred = model.predict(x_test)
            TestDataset[score_variable] = y_pred
            ###############################################################################

        elif ml_algorithm=='RFREG':
            ###############################################################################
            import sklearn
            model_attributes['MLTool'] = 'sklearn={}'.format(sklearn.__version__) 
            # Random Forest Model
            model, summary, model_fit_time= build_regression_model(x_train, y_train, model_variables, target_variable, model_parameters)
            model_interpretation['ModelSummary'] = summary  
            
    #        y_pred_prob = model.predict(x_validate)
    #        ValidateDataset[score_variable]=y_pred_prob
            
            y_pred = model.predict(x_test)
            TestDataset[score_variable] = y_pred
            ###############################################################################
        else:
            raise Exception('No Regression MLAlgorithm is specified in the model_parameters!')

        

    
    model_attributes['ModelFitTime'] = model_fit_time
    ############################################################################### 
    if ml_algorithm[-3:]=='REG':
        model_evaluation['RobustnessTable'], model_evaluation['ROCCurve'], model_evaluation['PrecisionRecallCurve'], model_evaluation['ROC_AUC'], model_evaluation['PRC_AUC'] = model_performance_matrics(TestDataset, target_variable=target_variable, score_variable=score_variable, quantile_label=quantile_label,  quantiles=quantiles, problem_type='regression', show_plot=show_plot)  
        model_evaluation['RMSE'], model_evaluation['R2'] = get_regression_errors(TestDataset, target_variable=target_variable, predicted_variable=score_variable)
    else:
        model_evaluation['RobustnessTable'], model_evaluation['ROCCurve'], model_evaluation['PrecisionRecallCurve'], model_evaluation['ROC_AUC'], model_evaluation['PRC_AUC'] = model_performance_matrics(TestDataset, target_variable=target_variable, score_variable=score_variable, quantile_label=quantile_label,  quantiles=quantiles, problem_type='classification', show_plot=show_plot)        

    if show_results:
        print(model_evaluation['RobustnessTable'])
    ###############################################################################
    
    ###############################################################################
    if return_model_object:
        Model = MLModel(model_attributes, model_parameters, sample_attributes, model_variables, variable_setup, 
                        target_variable, score_parameters, model_interpretation, model_evaluation, model_object=model)
    else:
        Model = MLModel(model_attributes, model_parameters, sample_attributes, model_variables, variable_setup, 
                        target_variable, score_parameters, model_interpretation, model_evaluation, model_object=None)
    ###############################################################################   
    # Cleanup memeory
    gc.collect()
    
    #return copy.deepcopy(Model)
    return Model

def model_guages_to_row(Model):
    guages = [{'Model':Model.get_model_id(),
                'ROC_AUC':Model.get_auc(curve='roc'),
                'PRC_AUC':Model.get_auc(curve='prc'),
            }]
    return pd.DataFrame(guages)
    
def model_guages_comparison(Models):
    model_guages = pd.DataFrame()
    for i in range(len(Models)):
        Model = Models[i]
        model_guages = model_guages.append(model_guages_to_row(Model), ignore_index=True)
    return model_guages
	
def confusion_matrix_comparison(DataFrame, Models, thresholds=0.5, score_variable=None, save_prediction=False, show_plot=False):
    """
    Parameters
    ----------
    DataFrame : pandas.DataFrame
        DataFrame
    Models : 
    thresholds : int or list(int), default 0.5    
    score_variable : str, optional, dafault None
        Name of the variable where the score is based on.  If None, score variabel is assigned by the model      
    save_prediction: bool, default False
        Retains prediction in column named by Model ID
    show_plot : bool, default False
        plot results if True

    Returns
    -------
    ConfusionMatrixComparison : pandas.DataFrame
    """
    try:
        if len(thresholds)>0:
            thresholds = thresholds        
    except:
        thresholds = [thresholds]
    
    ConfusionMatrixComparison=pd.DataFrame()
    
    for i in range(len(Models)):
        
        Model = Models[i]
        if score_variable==None:
            score_variable = Model.get_score_variable()
            
        DataFrame=score_processed_dataset(DataFrame, Model, edges=None, score_label=None, fill_missing=0)
        
        for threshold in thresholds:            
            model_id = Model.get_model_id()+'_[TH={}]'.format(threshold)    
            target_variable = Model.get_target_variable()
            predict_column = 'Predict'
            DataFrame[predict_column] = np.where(DataFrame[score_variable]>threshold,1,0)
            ConfusionMatrix=confusion_matrix(DataFrame, actual_variable=target_variable, predcted_variable=predict_column, labels=[0,1], sample_weight=None, totals=True)
            ConfusionMatrixComparison=ConfusionMatrixComparison.append(confusion_matrix_to_row(ConfusionMatrix, ModelID=model_id),ignore_index=True) 
            if save_prediction:
                pass
            else:
                DataFrame.drop(columns=[predict_column], inplace=True)
                
    if show_plot:
        ConfusionMatrixComparison[['PPV', 'TPR', 'ACC', 'F1']].plot(kind='bar', subplots=True, figsize=(5,10), legend=False)
    
    return ConfusionMatrixComparison

###############################################################################
##[ JSON INTERFACE ]###########################################################
###############################################################################
    
def build_ml_model_task(DataFrame, model_setup_dict, variables_setup_dict, return_script=False):
    """
    Parameters
    ----------
    DataFrame: pandas.DataFrame
    model_setup_dict: json or dict
    variables_setup_dict: json or dict
    
    Returns
    -------
    Model: mltk.MLModel
        
    """    
    import json
    import pandas as pd
    
    if type(model_setup_dict)==dict:
        pass
    else:
        try:
            model_setup_dict = json.loads(model_setup_dict) 
        except:
            print('ERROR in fitting model:{}\n {}'.format(model_setup_dict, traceback.format_exc()))  

    if type(variables_setup_dict)==dict:
        pass
    else:
        try:
            variables_setup_dict = json.loads(variables_setup_dict) 
        except:
            print('ERROR in fitting model:{}\n {}'.format(variables_setup_dict, traceback.format_exc()))  
    
    if type(DataFrame)==dict:
        dict_keys = DataFrame.keys()
        if ('TrainDataset' in dict_keys) and ('ValidateDataset' in dict_keys) and ('TestDataset' in dict_keys):
            TrainDataset = DataFrame['TrainDataset']
            ValidateDataset = DataFrame['ValidateDataset']
            TestDataset = DataFrame['TestDataset']
        else:
            print('Data input error check if the input dictionary has keys: "TrainDataset",  "ValidateDataset", "TestDataset" ')
    else:    
        TrainDataset, ValidateDataset, TestDataset = train_validate_test_split(DataFrame, ratios=model_setup_dict['sample_split'])
    
    model_data_stats = data_description(TrainDataset)
    model_setup_dict['sample_attributes']['ModelDataStats'] = model_data_stats
    
    TrainDataset, category_variables, binary_variables, target_variable, preprocess_variables_script = setup_variables_task(TrainDataset, variables_setup_dict, return_script=True)
    TrainDataset, feature_variables, target_variable = to_one_hot_encode(TrainDataset, 
                                                                         category_variables=category_variables, 
                                                                         binary_variables=binary_variables, 
                                                                         target_variable=target_variable)
    
    ValidateDataset, category_variables_, binary_variables_, target_variable_ = setup_variables_task(ValidateDataset, variables_setup_dict)
    ValidateDataset, feature_variables, target_variable = to_one_hot_encode(ValidateDataset, 
                                                                         category_variables=category_variables, 
                                                                         binary_variables=binary_variables, 
                                                                         target_variable=target_variable)
    
    TestDataset, category_variables_, binary_variables_, target_variable_  = setup_variables_task(TestDataset, variables_setup_dict)
    TestDataset, feature_variables, target_variable = to_one_hot_encode(TestDataset, 
                                                                         category_variables=category_variables, 
                                                                         binary_variables=binary_variables, 
                                                                         target_variable=target_variable)
        
    Model = build_ml_model(TrainDataset, ValidateDataset, TestDataset, 
                                  model_variables=model_setup_dict['model_variables'],
                                  variable_setup = variables_setup_dict,
                                  target_variable=model_setup_dict['target_variable'],
                                  model_attributes=model_setup_dict['model_attributes'], 
                                  sample_attributes=model_setup_dict['sample_attributes'], 
                                  model_parameters=model_setup_dict['model_parameters'], 
                                  score_parameters=model_setup_dict['score_parameters'], 
                                  return_model_object=model_setup_dict['model_outputs']['return_model_object'], 
                                  show_results=model_setup_dict['model_outputs']['show_results'], 
                                  show_plot=model_setup_dict['model_outputs']['show_plot']
                                  )
    

    if return_script:
        variables_setup_dict = {
            "setting":"score",    
            "variables": variables_setup_dict["variables"],    
            "preprocess_tasks": preprocess_variables_script    
        }
        return Model, variables_setup_dict
    else:
        return Model

def build_ml_model_from_data_task(DataFrame, model_building_setup_dict, return_script=False):
    """
    Parameters
    ----------
    DataFrame: pandas.DataFrame
    model_building_setup_dict: json or dict
    
    Returns
    -------
    Model: mltk.MLModel
    """   
    
    import json
    
    if type(model_building_setup_dict)==dict:
        pass
    else:
        try:
            model_building_setup_dict = json.loads(model_building_setup_dict) 
        except:
            print('ERROR in fitting model:{}\n {}'.format(model_building_setup_dict, traceback.format_exc())) 
            
    model_setup_dict = model_building_setup_dict['model_setup_dict']
    variables_setup_dict = model_building_setup_dict['variables_setup_dict']
    
    Model, variables_setup_dict = build_ml_model_task(DataFrame, model_setup_dict=model_setup_dict, variables_setup_dict=variables_setup_dict, return_script=True)
    print(Model.model_attributes['ModelID'])
    print('ROC AUC: ', Model.get_auc(curve='roc'))
    print('PRC AUC: ', Model.get_auc(curve='prc'))
    
    if return_script:
        return Model, variables_setup_dict
    else:
        return Model
