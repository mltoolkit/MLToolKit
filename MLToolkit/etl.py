# -*- coding: utf-8 -*-
# MLToolkit (mltoolkit)

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

edges_std = ['0', '1p', '1n', '1u', '1m', '1c', '1', '100', '500', 
             '1K', '2K', '5K', '10K', '20K', '50K', '100K', '500K', 
             '1M', '2M', '5M', '10M', '100M', '200M', '500M', 
             '1G', '2G', '5G', '10G', '100G', '200G', '500G',
             '1T', '2T', '5T', '10T', '100T', '200T', '500T',
             '1P', '2P', '5P', '10P', '100P', '200P', '500P',
             '1E']

def num_label_to_value(num_label):
    units = {'p':0.000000000001,
        'n':0.000000001,
        'u':0.000001,
        'm':0.001,
        'c':0.01,
        'd':0.1,
        '':1,
        'D':10,
        'H':100,
        'K':1000,
        'M':1000000,
        'G':1000000000,
        'T':1000000000000,
        'P':1000000000000000,
        'E':1000000000000000000,
        'INF':np.inf        
        }
    try:
        sign, inf, num, unit = re.findall('^([-]?)((\d+)([pnumcdDHKMGTPE]?)|INF)$', num_label.rstrip().lstrip())[0]
        if inf=='INF':
            value = int('{}1'.format(sign))*np.inf
        else:
            value = int('{}1'.format(sign))*float(num)*units[unit]
    except:
        print('vnum_label_value failed !\n{}'.format(traceback.format_exc()))
        value = None
    return value

def get_data_sql(query=None, server=None, database=None, uid=None, pwd=None):
    if query!=None and server!=None:        
        coerce_float=True
        index_col=None
        parse_dates=None
        
        try:
            if uid==None and pwd==None:
                connect_string = r'Driver={SQL Server};SERVER='+server+';DATABASE='+database+';TRUSTED_CONNECTION=yes;'
            else:
                connect_string = r'Driver={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+uid+'r;PWD='+pwd+'}'
            
            connection = pyodbc.connect(connect_string)        
            DataFrame = pd.read_sql_query(sql=query, con=connection, coerce_float=coerce_float, index_col=index_col, parse_dates=parse_dates)
            connection.close()        
        except:
            print('Database Query Fialed:\n{}\n'.format(traceback.format_exc()))
            DataFrame=pd.DataFrame()
    else:
        print('No Query provided !')
        DataFrame=pd.DataFrame()
       
    return DataFrame
	
def edge_labels_to_values(edge_labels, left_inclusive=False, right_inclusive=False):
    edge_values = []
    bin_labels = []
    n_bins = len(edge_labels)-1
    i=0
    for i in range(n_bins):        
        l_bracket = '(' if (i==0 and edge_labels[i]=='-INF') or (not left_inclusive) else '['
        r_bracket = ')' if (i==n_bins-1 and edge_labels[i+1]=='INF') or (not right_inclusive) else ']'
        edge_values.append(num_label_to_value(edge_labels[i]))
        bin_labels.append('{}_{}{},{}{}'.format(i+1, l_bracket, edge_labels[i], edge_labels[i+1], r_bracket))
    edge_values.append(num_label_to_value(edge_labels[n_bins]))
    return edge_values,bin_labels
	
def add_missing_feature_columns(DataFrame, expected_features, fill_value=0):
    # Blanck columns for non-existance variables
    feature_variables_to_add = list(set(expected_features) - set(DataFrame.columns)) # Find columns not found in the dataset
    for f in feature_variables_to_add:
        DataFrame[f]=fill_value
        print('Column [{}] does not exist in the dataset. Created new column and set to {}...'.format(f,fill_value))
    return DataFrame

def exclude_records(DataFrame, exclude_ondition=None, action = 'flag', exclude_label='_EXCLUDE_'):
    
    if exclude_ondition==None:
        print('No exclude condition...')
        return DataFrame
    
    try:
        if action=='drop': #Drop Excludes        
            DataFrame = DataFrame.query('not ({})'.format(exclude_ondition))
        elif action=='flag': #Create new flagged column
            DataFrame[exclude_label] = DataFrame.eval(exclude_ondition).astype('int8')
            print('Records {} -> {}=1'.format(exclude_ondition, exclude_label))
    except:
        print('Error in excluding records {}:\n{}\n'.format(exclude_ondition, traceback.format_exc()))
        
    return DataFrame

def set_binary_target(DataFrame, target_condition=None, target_variable='_TARGET_'):
    if target_condition==None: 
        return DataFrame
    
    try:        
        DataFrame[target_variable] = DataFrame.eval(target_condition).astype('int8')
    except:
        print('Error in creating the target variable {}:\n{}\n'.format(target_condition, traceback.format_exc()))
        
    return DataFrame  

def numeric_to_category(DataFrame, variable, str_labels, right_inclusive=True, print_output=False, return_variable=False):
    edge_values, bin_labels = edge_labels_to_values(str_labels, left_inclusive=not right_inclusive, right_inclusive=right_inclusive)
    group_variable = '{}GRP'.format(variable)
    DataFrame[group_variable] = pd.cut(DataFrame[variable], bins=edge_values, labels=bin_labels, right=right_inclusive, include_lowest=True)
    
    if print_output:
        print(DataFrame.groupby(by=group_variable)[group_variable].count())
        
    if return_variable:
        return DataFrame, group_variable
    else:
        return DataFrame
    
def variable_to_binary(DataFrame, str_condition, group_variable=None, fill_missing=0, print_output=False, return_variable=False):
    if group_variable==None:
        group_variable = '{}'.format(str_condition)
    else:
        group_variable = group_variable
        
    DataFrame[group_variable] = DataFrame.eval(str_condition).astype('int8').fillna(fill_missing)

        
    if print_output:
        print(DataFrame.groupby(by=group_variable)[group_variable].count())
        
    if return_variable:
        return DataFrame, group_variable
    else:
        return DataFrame