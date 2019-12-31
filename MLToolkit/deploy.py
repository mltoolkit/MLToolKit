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
- Data Extraction (SQL, Flatfiles, etc.)
- Exploratory data analysis (statistical summary, univariate analysis, etc.)
- Feature Extraction and Engineering
- Model performance analysis and comparison between models
- Hyper parameter tuning
- Auto ML (automated machine learning)
- Serving models via RESTful  API

Author
------
- Sumudu Tennakoon

Links
-----
Website: http://sumudu.tennakoon.net/projects/MLToolkit
Github: https://github.com/sptennak/MLToolkit

License
-------
Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""

from datetime import datetime
import gc
import traceback
import gc
import os
from timeit import default_timer as timer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import warnings
warnings.filterwarnings("ignore")

def score(DataFrame, score_variable='Probability', edges=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], score_label='Score'):
    score=np.arange(1, len(edges), 1)
    #Ref: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
    DataFrame[score_label]=pd.cut(DataFrame[score_variable], bins=edges, labels=score, include_lowest=True, right=True).astype('int8')          
    return DataFrame

def set_predicted_columns(DataFrame, score_variable, threshold=0.5, fill_missing=0):
    """
    Parameters
    ----------
    DataFrame : pandas.DataFrame
        DataFrame
    score_variable : str
        Name of the variable where the score is based on.    
    thresholds : int or list(int), default 0.5    
    fill_missing : int, default 0

    Returns
    -------
    DataFrame : pandas.DataFrame
    """
    str_condition = '{}>{}'.format(score_variable, threshold)
    DataFrame['Predicted'] = DataFrame.eval(str_condition).astype('int8').fillna(fill_missing)    
    #np.where(TestDataset[score_variable]>threshold,1,0)
    return DataFrame
          
def score_processed_dataset(DataFrame, Model, edges=None, score_label=None, fill_missing=0, verbose=False):    
    target_variable = Model.get_target_variable()
    model_variables = Model.get_model_variables()
    score_variable = Model.get_score_variable()   
    
    if edges==None:
        edges=Model.get_score_parameter('Edges')
    
    if score_label==None:
        score_label=Model.get_score_parameter('ScoreLabel')
        
    # Blanck columns for non-existance variables
    missing_variables = list(set(model_variables) - set(DataFrame.columns)) # Find columns not found in the dataset
    for f in missing_variables:
        DataFrame[f]=fill_missing
        if verbose:
            print('Column [{}] does not exist in the dataset. Created new column and set to {}...'.format(f,missing_variables))
        
    x_test = DataFrame[model_variables].values
    if verbose:
        print('Test Samples: {} loded...'.format(x_test.shape[0]))
    
    ml_algorithm = Model.model_parameters['MLAlgorithm']    
    if ml_algorithm=='LGR':
        y_pred_prob = Model.model_object.predict(x_test)
    elif ml_algorithm=='RF':        
        y_pred_prob = Model.model_object.predict_proba(x_test)[:,1]
    elif ml_algorithm=='NN':     
        batch_size=Model.model_parameters['BatchSize']
        y_pred_prob = Model.model_object.predict(x_test, verbose=1, batch_size=batch_size)[:,1]
        
    DataFrame[score_variable] = y_pred_prob
    DataFrame=score(DataFrame, score_variable=score_variable, edges=edges, score_label=score_label)
    
    return DataFrame
	
def score_records(record, Model, edges=None, ETL=None, return_type='frame', json_orient='records'):
    """
    Parameters
    ----------
    record : pandas.DataFrame or JSON str
        Can input multiple records in JSON string in records format.    
        E.g.: ['[{"ID":1, "age":32", hoursperweek":40} ,  {"ID":2, "age":28", hoursperweek":40}]    
    Model : mltk.MLModel 
        MLModel object
    edges : list(float), default None
        Bin edges for scoring. E.g.: [0.0, 0.1, 0.2, 0.3, 0.4, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ETL: function, defualt None
        Input data pre processing function. Will not continue scoring if not provided. 
        Function has to be created in the form below, returning a pandas.DataFrame as output.
        def ETL(InputDataFrame):
            ...
            ...
            return OutputDataFrame
    return_type : {'frame', 'json, 'dict'}, default 'frame'
        'frame' -> pandas.DataFrame
        'json -> JSON str
        'dict' -> dict
    json_orient: {'records', 'split'}, default 'records'
        Expected JSON string format in the input and outputs if used.
        'records' : [{column -> value}, ... , {column -> value}]
        'split' : {index -> [index], columns -> [columns], data -> [values]}
    Returns
    -------
    ScoreDataset : {pandas.DataFrame, JSON str, dict}
        Output type based on the parameter return_type
    """
    if ETL==None:
        print('No ETL function provided')
        return None
    
    if type(record)==pd.core.frame.DataFrame:
        ScoreDataset=record
    else:    
        try:
            import json
            try:
                ScoreDataset = pd.read_json(record, orient=json_orient)
            except:
                ScoreDataset = pd.read_json('[{}]'.format(record), orient=json_orient)
        except:
            print('Input data parsing error: \n{}\n'.format(traceback.format_exc()))
            return None
    
    score_label = Model.get_score_label()
    score_variable = Model.get_score_variable()
    #input_columns = list(ScoreDataset.columns.values) #will not work if column names were cleaned
    result_columns = [score_variable,score_label]
    ScoreDataset, input_columns = ETL(ScoreDataset)
    ScoreDataset = score_processed_dataset(ScoreDataset, Model, edges=edges, score_label=None, fill_missing=0)
    
    if return_type=='json':
        return ScoreDataset[input_columns+result_columns].to_json(orient=json_orient)
    elif return_type=='dict':
        return ScoreDataset[input_columns+result_columns].to_dict(orient=json_orient)
    elif return_type=='frame':
        return ScoreDataset[input_columns+result_columns]
    else:
        return ScoreDataset[score_label].values[0]      