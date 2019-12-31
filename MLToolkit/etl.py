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
import pyodbc
from timeit import default_timer as timer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import warnings
warnings.filterwarnings("ignore")

def remove_special_characters(str_val, replace=''):
    return re.sub('\W+',replace, str_val)

def remove_special_characters_list(str_list, replace=''):
    return [remove_special_characters(str_val, replace=replace) for str_val in str_list]

def check_list_values_unique(values_list):
    if len(values_list) == len(set(values_list)):
        return True
    else:
        return False
    
def clean_column_names(DataFrame, replace=''): # Remove special charcters from column names
    """
    Parameters
    ----------
    DataFrame : pandas.DataFrame
        DataFrame
    replace : str, dafault ''
        Character to replace special charaters with.    
    
    Returns
    -------
    DataFrame : pandas.DataFrame
    """
    try:
        columns = DataFrame.columns
        columns = remove_special_characters_list(columns, replace=replace)
        if check_list_values_unique(columns):
            DataFrame.columns = columns
        else:
            print('Duplicates values excists the column names after removing special characters!. Column names were rolled-back to initial values.')        
    except:
        print('Error in removing special characters from column names:\n{}\n'.format(traceback.format_exc()))
    return DataFrame

def handle_duplicate_columns(DataFrame, action='rename'): #'drop'
    """
    Parameters
    ----------
    DataFrame : pandas.DataFrame
        DataFrame
    action : {'rename', 'drop'}, dafault 'rename'
        Action to be taken on duplicate columns    
    
    Returns
    -------
    DataFrame : pandas.DataFrame
    """
    is_duplicate = DataFrame.columns.duplicated()
    columns = list(DataFrame.columns)
    if action=='rename':
        for i in range(len(columns)):
            if is_duplicate[i]:
               columns[i]=columns[i]+'_' 
        DataFrame.columns = columns
    elif action=='drop':
        DataFrame = DataFrame.loc[:,~is_duplicate]
    else:
        print('No valid action (rename or drop) provided!')
    return DataFrame

def merge_categories(DataFrame, groups):
    """
    Parameters
    ----------
    DataFrame : pandas.DataFrame
        DataFrame
    groups :  [{},]
        e.g. groups = [{'variable':'grade', 'group_name':'A', 'values':['A+', 'A', A-']},] 
    
    Returns
    -------
    DataFrame : pandas.DataFrame
    """
    for group in groups:
        try:
            DataFrame[group['variable']].replace(to_replace=group['values'], value=group['group_name'], inplace=True)
        except:
            print('ERROR merging categories of variable {}: {}'.format(group['variable'], traceback.format_exc()))  
    return DataFrame
   
def create_binary_variables_set(DataFrame, conditions, return_variable=False): #e.g. conditions = [{'bin_variable':'CapitalGainPositive', 'str_condition':"capitalgain>0"},]
    """
    Parameters
    ----------
    DataFrame : pandas.DataFrame
        DataFrame
    conditions :  [{},]
        e.g. conditions = [{'bin_variable':'CapitalGainPositive', 'str_condition':"capitalgain>0"},] 
    return_variable: bool, default False
    
    Returns
    -------
    DataFrame : pandas.DataFrame
    """
    bin_variables = []
    for condition in conditions:
        try:
            DataFrame, group_variable = variable_to_binary(DataFrame, condition['str_condition'], group_variable=condition['bin_variable'], fill_missing=0, print_output=False, return_variable=True)
            bin_variables.append(group_variable)
        except:
            print('ERROR creating bunary variable {} = {} to buckets: {}'.format(condition['bin_variable'], condition['str_condition'], traceback.format_exc()))  

    if return_variable:
        return DataFrame, bin_variables
    else:
        return DataFrame

def create_categorical_variables_set(DataFrame, buckets, return_variable=False): #e.g. buckets =[{'variable':'age', 'str_labels':['0', '20', '30', '40', '50', '60', 'INF']},]
    """
    Parameters
    ----------
    DataFrame : pandas.DataFrame
        DataFrame
    buckets :  list(dict) [{},]
        e.g. buckets =[{'variable':'age', 'str_labels':['0', '20', '30', '40', '50', '60', 'INF']},]
    return_variable: bool, default False
    
    Returns
    -------
    DataFrame : pandas.DataFrame
    """
    category_variables = []
    for bucket in buckets:
        try:
            DataFrame, group_variable = numeric_to_category(DataFrame=DataFrame, variable=bucket['variable'], str_labels=bucket['str_labels'], right_inclusive=True, print_output=False,  return_variable=True)
            category_variables.append(group_variable)
        except:
            print('ERROR convering {} to buckets: {}'.format(bucket['variable'], traceback.format_exc()))

    if return_variable:
        return DataFrame, category_variables
    else:
        return DataFrame
    
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

def get_data_sql(query=None, server=None, database=None, auth=None, uid=None, pwd=None):
    """
    Parameters
    ----------
    query : str
        SQL SELECT query
    server : str
        Database Server
    database : str
        Database
    auth :  dict
        e.g. auth = {'type':'user', 'uid':'user', 'pwd':'password'} for username password authentication
             auth = {'type':'machine', 'uid':None, 'pwd':None} for machine authentication
    return_variable: bool, default False
    
    Returns
    -------
    DataFrame : pandas.DataFrame
    """    
    if query!=None and server!=None and auth!=None:        
        coerce_float=True
        index_col=None
        parse_dates=None
        
        try:
            if auth['type']=='machine':
                connect_string = r'Driver={SQL Server};SERVER='+server+';DATABASE='+database+';TRUSTED_CONNECTION=yes;'
            elif auth['type']=='user':
                connect_string = r'Driver={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+uid+'r;PWD='+pwd+'}'
            else:
                raise Exception('No db server authentication method provided!')
            connection = pyodbc.connect(connect_string)        
            DataFrame = pd.read_sql_query(sql=query, con=connection, coerce_float=coerce_float, index_col=index_col, parse_dates=parse_dates)
            connection.close()        
        except:
            print('Database Query Fialed!:\n{}\n'.format(traceback.format_exc()))
            DataFrame=pd.DataFrame()
    else:
        print('No Query provided !')
        DataFrame=pd.DataFrame()
       
    return DataFrame
	
def edge_labels_to_values(edge_labels, left_inclusive=False, right_inclusive=False):
    """
    Parameters
    ----------
    edge_labels : str []
        Edge labels with number unit as postfix
            'p':0.000000000001,
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
            'INF':np.inf        
    left_inclusive : bool, default False
        Include left edge
    right_inclusive : bool, default False
        Include right edge
    
    Returns
    -------
    edge_values : numeric []
    bin_labels : str []
    """ 
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
    N0 = len(DataFrame.index)
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
    N1 = len(DataFrame.index)    
    print('{} records were excluded'.format(N1-N0))
    return DataFrame

def set_binary_target(DataFrame, target_condition=None, target_variable='_TARGET_'):
    if target_condition==None: 
        return DataFrame
    
    try:        
        DataFrame[target_variable] = DataFrame.eval(target_condition).astype('int8')
    except:
        print('Error in creating the target variable {}:\n{}\n'.format(target_condition, traceback.format_exc()))
        
    return DataFrame  

def numeric_to_category(DataFrame, variable, str_labels, group_variable=None, right_inclusive=True, print_output=False, return_variable=False):
    """
    Parameters
    ----------
    DataFrame : pandas.DataFrame
        DataFrame
    variable : str
        Numeric variable to categorize
    str_labels : str []
        Edge labels with number unit as postfix
            'p':0.000000000001,
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
            'INF':np.inf        
    group_variable: str, optional, default None
        Name of the new variable.
    right_inclusive : bool, default True
        Include right edge. Activates right_inclusive=True if False
    print_output : bool, default False
        Print result
    return_variable : bool, default False
        Return categorical variable name        
    Returns
    -------
    DataFrame : pandas.DataFrame
    group_variable : str
    """
    edge_values, bin_labels = edge_labels_to_values(str_labels, left_inclusive=not right_inclusive, right_inclusive=right_inclusive)

    if group_variable==None:
        group_variable = '{}GRP'.format(variable)
        
    DataFrame[group_variable] = pd.cut(DataFrame[variable], bins=edge_values, labels=bin_labels, right=right_inclusive, include_lowest=True)
    
    if print_output:
        print(DataFrame.groupby(by=group_variable)[group_variable].count())
        
    if return_variable:
        return DataFrame, group_variable
    else:
        return DataFrame
    
def variable_to_binary(DataFrame, str_condition, group_variable=None, fill_missing=0, print_output=False, return_variable=False):
    """
    Parameters
    ----------
    DataFrame : pandas.DataFrame
        DataFrame
    variable : str
        Numeric variable to categorize
    str_condition : str
        Conditional statemnt.
    group_variable: str, optional, default None
        Name of the new variable.
    fill_missing : int8, default 0
        Value to fill missing or NaN
    print_output : bool, default False
        Print result
    return_variable : bool, default False
        Return categorical variable name.       
    Returns
    -------
    DataFrame : pandas.DataFrame
    group_variable : str
    """
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
    
def setup_variables(DataFrame, target_variable, category_variables=[], binary_variables=[], conditions=[], buckets=[], groups=[]):
    """
    Parameters
    ----------
    DataFrame : pandas.DataFrame
        DataFrame
    target_variable : str
    category_variables : list, default []
    binary_variables : list, default []
    conditions: list(dict), default []
    buckets : list(dict), default []
    groups : list(dict), default []
      
    Returns
    -------
    DataFrame : pandas.DataFrame
    category_variables : list
    binary_variables : list
    target_variable :list
    """
    category_variables =  set(category_variables) & set(DataFrame.columns)
    binary_variables  = set(binary_variables) & set(DataFrame.columns)
    
    # Check if target variable exists (fill the column with None in scoring)
    if not target_variable in DataFrame.columns:
        DataFrame[target_variable]=None    
    
    # Create more Binary variables
    DataFrame, bin_variables_ = create_binary_variables_set(DataFrame, conditions, return_variable=True)
    binary_variables.update(bin_variables_)
                  
    # Create more Catergorical variables
    DataFrame, category_variables_ = create_categorical_variables_set(DataFrame, buckets, return_variable=True)
    category_variables.update(category_variables_)

    # Merge categorical values
    DataFrame = merge_categories(DataFrame, groups)
    
    #Finalize variables lists
    category_variables=list(category_variables)
    binary_variables=list(binary_variables)
    target_variable = target_variable
    
    return DataFrame, category_variables, binary_variables, target_variable