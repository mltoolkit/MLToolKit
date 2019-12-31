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
try:
    import pyodbc
except:
    print('pyodbc not found! Data base query fufnctions disabled.')
import warnings
warnings.filterwarnings("ignore")

from mltk.string import *

edges_std = ['0', '1p', '1n', '1u', '1m', '1c', '1', '100', '500', 
             '1K', '2K', '5K', '10K', '20K', '50K', '100K', '500K', 
             '1M', '2M', '5M', '10M', '100M', '200M', '500M', 
             '1G', '2G', '5G', '10G', '100G', '200G', '500G',
             '1T', '2T', '5T', '10T', '100T', '200T', '500T',
             '1P', '2P', '5P', '10P', '100P', '200P', '500P',
             '1E']

###############################################################################
##[ I/O FUNCTIONS]#############################################################      
###############################################################################

def read_data_csv(file, separator=',', encoding=None):
    return pd.read_csv(filepath_or_buffer=file, sep=separator, encoding=encoding)

def read_data_pickle(file, compression='infer'):
    return pd.read_csv(filepath_or_buffer=file, compression=compression)
    
def read_data_sql(query=None, server=None, database=None, auth=None):
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
                uid =  auth['uid'] 
                pwd =  auth['pwd'] 
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

def write_data_sql(DataFrame, server=None, database=None, schema=None, table=None, dtypes=None, if_exists='fail', auth=None):
    """
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_sql.html
    
    Parameters
    ----------
    DataFrame : pandas.DataFrame
        DataFrame
    server : str
        Database Server
    database : str
        Database
    schema : str
        Database Schema
    table : str
        Table name
    if_exists : {'fail', 'replace', 'append'}, default 'fail'
        Action if the table already exists.
    auth :  dict
        e.g. auth = {'type':'user', 'uid':'user', 'pwd':'password'} for username password authentication
             auth = {'type':'machine', 'uid':None, 'pwd':None} for machine authentication
    
    Returns
    -------
    None
    """    
    if server!=None and  database!=None and schema!=None and table!=None and auth!=None : 
        try:
            if auth['type']=='machine':
                connect_string = r'Driver={SQL Server};SERVER='+server+';DATABASE='+database+';TRUSTED_CONNECTION=yes;'
            elif auth['type']=='user':
                uid =  auth['uid'] 
                pwd =  auth['pwd'] 
                connect_string = r'Driver={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+uid+'r;PWD='+pwd+'}'
            else:
                raise Exception('No db server authentication method provided!')
                
            connection = pyodbc.connect(connect_string)  
            if dtypes==None:
                DataFrame.to_sql_query(name=table, con=connection, schema=schema, if_exists=if_exists)
            else:
                DataFrame.to_sql_query(name=table, con=connection, schema=schema, dtype=dtypes, if_exists=if_exists)
            connection.close()             
        except:
            print('Database Query Fialed!:\n{}\n'.format(traceback.format_exc()))
            DataFrame=pd.DataFrame()
    else:
        print('Check the destiniation table path (server, database, schema, table, auth) !')
        DataFrame=pd.DataFrame()        

###############################################################################
##[ VALIDATE FIELDS]##########################################################      
###############################################################################
        
def add_identity_column(DataFrame, id_label='ID', start=1, increment=1):
    if id_label in DataFrame.columns:
        print('Column {} exists in the DataFrame'.format(id_label))
        return DataFrame
    else:
        DataFrame.reset_index(drop=True, inplace=True)
        DataFrame.insert(0, id_label, start+DataFrame.index)
        return DataFrame
    
def remove_special_characters(str_val, replace=''):
    return re.sub('\W+',replace, str_val)

def remove_special_characters_list(str_list, replace=''):
    return [remove_special_characters(str_val, replace=replace) for str_val in str_list]
    
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

def check_list_values_unique(values_list):
    if len(values_list) == len(set(values_list)):
        return True
    else:
        return False
    
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

def add_missing_feature_columns(DataFrame, expected_features, fill_value=0):
    # Blanck columns for non-existance variables
    feature_variables_to_add = list(set(expected_features) - set(DataFrame.columns)) # Find columns not found in the dataset
    for f in feature_variables_to_add:
        DataFrame[f]=fill_value
        print('Column [{}] does not exist in the dataset. Created new column and set to {}...'.format(f,fill_value))
    return DataFrame

def exclude_records(DataFrame, exclude_condition=None, action = 'flag', exclude_label='_EXCLUDE_'):
    N0 = len(DataFrame.index)
    if exclude_condition==None:
        print('No exclude condition...')
        return DataFrame
    
    try:
        if action=='drop': #Drop Excludes        
            DataFrame = DataFrame.query('not ({})'.format(exclude_condition))
        elif action=='flag': #Create new flagged column
            DataFrame[exclude_label] = DataFrame.eval(exclude_condition).astype('int8')
            print('Records {} -> {}=1'.format(exclude_condition, exclude_label))
    except:
        print('Error in excluding records {}:\n{}\n'.format(exclude_condition, traceback.format_exc()))
    N1 = len(DataFrame.index)    
    print('{} records were excluded'.format(N1-N0))
    return DataFrame

###############################################################################
##[ CREATING FEATURES - TRANSFORMATIONS]#######################################      
############################################################################### 
    
def normalize_variable(DataFrame, variable, method='maxscale', parameters=None, to_variable=None):
    """
    Reference: https://en.wikipedia.org/wiki/Normalization_(statistics)
    
    Parameters
    ----------
    DataFrame : pandas.DataFrame
        DataFrame
    variable : str
        Numeric variable to normalize
    method : {'minscale', 'maxscale', 'range', 'zscore', 'mean', 'median'}, default 'maxscale'
        'minscale' : value/min
        'maxscale' : value/max
        'range' : value/abs(max-min)
        'minmaxfs' : (value-min)/(max-min)
            Min-Max Feature scaling
        'minmaxfs_m' : (value-mean)/(max-min)
            Mean normalization
        'zscore' : (value-mean)/std
            Standardization (Z-score Normalization)
        'mean' : value/mean
        'median' : value/median
        
    Returns
    -------
    DataFrame : pandas.DataFrame
    """ 
    if to_variable==None:
        to_variable = variable
        
    if method=='minscale': #scale=max
        DataFrame[to_variable] = DataFrame[variable]/DataFrame[variable].min()
    if method=='maxscale': #scale=max
        DataFrame[to_variable] = DataFrame[variable]/DataFrame[variable].max()
    if method=='range': # range = abs(max-min)
        min_max = abs(DataFrame[variable].max()-DataFrame[variable].min())
        DataFrame[to_variable] = DataFrame[variable]/min_max
    if method=='minmaxfs': # range = (value-min)/(max-min)
        min_=DataFrame[variable].min()
        max_=DataFrame[variable].max()
        min_max = abs(max_-min_)
        DataFrame[to_variable] = (DataFrame[variable]-min_)/min_max
    if method=='minmaxfs_m': # range = (value-min)/(max-min)
        min_=DataFrame[variable].min()
        max_=DataFrame[variable].max()
        mean = DataFrame[variable].mean()
        min_max = abs(max_-min_)
        DataFrame[to_variable] = (DataFrame[variable]-mean)/min_max
    if method=='mean':
        mean = DataFrame[variable].mean()
        DataFrame[to_variable] = DataFrame[variable]/mean
    if method=='median':
        median = DataFrame[variable].median()
        DataFrame[to_variable] = DataFrame[variable]/median
    if method=='zscore':  
        std = DataFrame[variable].std()
        mean = DataFrame[variable].mean()
        DataFrame[to_variable] = (DataFrame[variable] - mean)/std  
        
    return DataFrame

def transform_variable(DataFrame, variable, to_variable, operation, parameters, return_variable=False):
    if operation=='normalize':
        method = parameters['method']
        DataFrame = normalize_variable(DataFrame, variable, method=method, to_variable=to_variable)
    else:
        pass # other transformations to be implemented
        
    if return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame
    
def create_transformed_variables(DataFrame, variable_operations, return_variables=False): #variable_operations = [{'variable':'age', 'to': 'normalized_age', 'operation':'normalize', 'parameters':{'method':'zscore'}},]
    transformed_variables = []
    for variable_operation in variable_operations:
        try:
            DataFrame, transformed_variable = transform_variable(DataFrame, variable_operation['variable'], variable_operation['to'], variable_operation['operation'], variable_operation['parameters'], return_variable=True)
            transformed_variables.append(transformed_variable)
        except:
            print('ERROR creating transformed variable for {} with {} : {}'.format(variable_operation['variable'], variable_operation['operation'], traceback.format_exc()))  
            
    if return_variables:
        return DataFrame, transformed_variables
    else:
        return DataFrame
    
def create_date_difference_variable(DataFrame, date_variable1, date_variable2, difference_variable=None, difference_unit='D', return_variable=False):
    if difference_variable==None:
        difference_variable = '{}DIFF{}'.format(date_variable2,date_variable1)
    else:
        difference_variable = difference_variable
    
    try:
        DataFrame[date_variable1] = pd.to_datetime(DataFrame[date_variable1])
        DataFrame[date_variable2] = pd.to_datetime(DataFrame[date_variable2])        
        DataFrame[difference_variable] = DataFrame[date_variable2] - DataFrame[date_variable1]
        DataFrame[difference_variable]=DataFrame[difference_variable]/np.timedelta64(1,difference_unit)
    except:
        DataFrame[difference_variable] = None
        print('Date Type Error: {}, {} '.format(date_variable1, date_variable2, traceback.format_exc()))  

    if return_variable:
        return DataFrame, difference_variable
    else:
        return DataFrame

def create_date_difference_variables(DataFrame, date_differences, return_variables=False):
    date_diffeence_variables = []
    
    for date_difference in date_differences:
        try:
            DataFrame, date_diffeence_variable = create_date_difference_variable(DataFrame, date_difference['date_variable1'], date_difference['date_variable2'], date_difference['difference_variable'], date_difference['difference_unit'], return_variable=True)
            date_diffeence_variables.append(date_diffeence_variable)            
        except:
            print('ERROR creating date difference variable between {} with {} : {}'.format(date_difference['date_variable1'], date_difference['date_variable2'], traceback.format_exc()))  
        
    if return_variables:
        return DataFrame, date_diffeence_variables
    else:
        return DataFrame    

def create_numeric_difference_variable(DataFrame, variable1, variable2, difference_variable=None, return_variable=False):
    if difference_variable==None:
        difference_variable = '{}DIFF{}'.format(variable1, variable2)
    else:
        difference_variable = difference_variable
    
    try:
        DataFrame[variable1] = pd.to_numeric(DataFrame[variable1], errors='coerce')
        DataFrame[variable2] = pd.to_numeric(DataFrame[variable2], errors='coerce')        
        DataFrame[difference_variable] = DataFrame[variable1] - DataFrame[variable2]
    except:
        DataFrame[difference_variable] = None
        print('Date Type Error: {}, {} '.format(variable1, variable2, traceback.format_exc()))  

    if return_variable:
        return DataFrame, difference_variable
    else:
        return DataFrame

def create_numeric_difference_variables(DataFrame, numeric_differences, return_variables=False):
    numeric_difference_variables = []
    
    for numeric_difference in numeric_differences:
        try:
            DataFrame, numeric_difference_variable = create_numeric_difference_variable(DataFrame, numeric_difference['date_variable1'], numeric_difference['date_variable2'], numeric_difference['difference_variable'], numeric_difference['difference_unit'], return_variable=True)
            numeric_difference_variables.append(numeric_difference_variable)            
        except:
            print('ERROR creating date difference variable between {} with {} : {}'.format(numeric_difference['date_variable1'], numeric_difference['date_variable2'], traceback.format_exc()))  
        
    if return_variables:
        return DataFrame, numeric_difference_variables
    else:
        return DataFrame     
    
###############################################################################
##[ CREATING FEATURES - CATEGORY]##############################################      
############################################################################### 
    
def merge_categories(DataFrame, merge_categories, return_variables=False):
    """
    Parameters
    ----------
    DataFrame : pandas.DataFrame
        DataFrame
    merge_categories :  [{},]
        e.g. merge_categories = [{'variable':'grade', category_variable='grade', 'group_value':'A', 'values':['A+', 'A', A-']},] 
    
    Returns
    -------
    DataFrame : pandas.DataFrame
    """
    category_variables = []
    
    for merge in merge_categories:
        try:
            DataFrame[merge['category_variable']] = DataFrame[merge['variable']].replace(to_replace=merge['values'], value=merge['group_value'])
            category_variables.append(merge['category_variable'])
        except:
            print('ERROR merging categories of variable {}: {}'.format(merge['variable'], traceback.format_exc()))  

    if return_variables:
        return DataFrame, category_variables
    else:
        return DataFrame

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

def set_binary_target(DataFrame, target_condition=None, target_variable='_TARGET_'):
    if target_condition==None: 
        return DataFrame
    
    try:        
        DataFrame[target_variable] = DataFrame.eval(target_condition).astype('int8')
    except:
        print('Error in creating the target variable {}:\n{}\n'.format(target_condition, traceback.format_exc()))
        print('Check variable setting !')
    return DataFrame  

def numeric_to_category(DataFrame, variable, str_labels, category_variable=None, right_inclusive=True, print_output=False, return_variable=False):
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
    category_variable: str, optional, default None
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
    category_variable : str
    """
    edge_values, bin_labels = edge_labels_to_values(str_labels, left_inclusive=not right_inclusive, right_inclusive=right_inclusive)

    if category_variable==None:
        category_variable = '{}GRP'.format(variable)
        
    DataFrame[category_variable] = pd.cut(DataFrame[variable], bins=edge_values, labels=bin_labels, right=right_inclusive, include_lowest=True)
    
    if print_output:
        print(DataFrame.groupby(by=category_variable)[category_variable].count())
        
    if return_variable:
        return DataFrame, category_variable
    else:
        return DataFrame

def create_categorical_variables(DataFrame, buckets, return_variables=False): #e.g. buckets =[{'category_variable':'AgeGRP', 'variable':'age', 'str_labels':['0', '20', '30', '40', '50', '60', 'INF']},]
    """
    Parameters
    ----------
    DataFrame : pandas.DataFrame
        DataFrame
    buckets :  list(dict) [{},]
        e.g. buckets =[{'variable':'age', 'category_variable':'ageGRP', 'str_labels':['0', '20', '30', '40', '50', '60', 'INF']},]
    return_variable: bool, default False
    
    Returns
    -------
    DataFrame : pandas.DataFrame
    """
    category_variables = []
    for bucket in buckets:
        try:
            DataFrame, category_variable = numeric_to_category(DataFrame=DataFrame, variable=bucket['variable'], str_labels=bucket['str_labels'], right_inclusive=bucket['right_inclusive'], print_output=False,  return_variable=True)
            category_variables.append(category_variable)
        except:
            print('ERROR convering {} to buckets: {}'.format(bucket['variable'], traceback.format_exc()))

    if return_variables:
        return DataFrame, category_variables
    else:
        return DataFrame
    
def variable_to_binary(DataFrame, str_condition, category_variable=None, fill_missing=0, print_output=False, return_variable=False):
    """
    Parameters
    ----------
    DataFrame : pandas.DataFrame
        DataFrame
    variable : str
        Numeric variable to categorize
    str_condition : str
        Conditional statemnt.
    category_variable: str, optional, default None
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
    category_variable : str
    """
    if category_variable==None:
        category_variable = '{}'.format(str_condition)
    else:
        category_variable = category_variable
        
    DataFrame[category_variable] = DataFrame.eval(str_condition).astype('int8').fillna(fill_missing)

        
    if print_output:
        print(DataFrame.groupby(by=category_variable)[category_variable].count())
        
    if return_variable:
        return DataFrame, category_variable
    else:
        return DataFrame

def create_binary_variables(DataFrame, conditions, return_variables=False): #e.g. conditions = [{'bin_variable':'CapitalGainPositive', 'str_condition':"capitalgain>0"},]
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
            DataFrame, category_variable = variable_to_binary(DataFrame, condition['str_condition'], category_variable=condition['bin_variable'], fill_missing=0, print_output=False, return_variable=True)
            bin_variables.append(category_variable)
        except:
            print('ERROR creating binary variable {} = {} to buckets: {}'.format(condition['bin_variable'], condition['str_condition'], traceback.format_exc()))  

    if return_variables:
        return DataFrame, bin_variables
    else:
        return DataFrame

def variable_pair_equality(DataFrame, variable1, variable2, category_variable, return_variable=False):
    DataFrame.loc[(DataFrame[variable1]==DataFrame[variable2]), category_variable] = 'EQ'
    DataFrame.loc[(DataFrame[variable1]!=DataFrame[variable2]), category_variable] = 'DF'
    DataFrame.loc[(DataFrame[variable1].isna()) | (DataFrame[variable2].isna()), category_variable] = 'ON'
    DataFrame.loc[(DataFrame[variable1].isna()) & (DataFrame[variable2].isna()), category_variable] = 'BN'
    
    if return_variable:
        return DataFrame, category_variable
    else:
        return DataFrame
    
def create_variable_pair_equality(DataFrame, variable_pairs, return_variables=False): #variable_pair = [{'variable1':'YearManufactured', 'variable2':'YearRegistered', 'category_variable':'Year'}]
    category_variables = []
    for variable_pair in variable_pairs:
        try:
            DataFrame, category_variable = variable_pair_equality(DataFrame, variable_pair['variable1'], variable_pair['variable2'], variable_pair['category_variable'], return_variable=True)
            category_variables.append(category_variable)
        except:
            print('ERROR creating equality variable for {} and {} : {}'.format(variable_pair['variable1'], variable_pair['variable2'], traceback.format_exc()))  

    if return_variables:
        return DataFrame, category_variables
    else:
        return DataFrame

def count_characters(DataFrame, variable, criteria='*', string_count_variable=None, return_variable=False):    
    if string_count_variable==None:
        string_count_variable = '{}CNT{}'.format(variable, remove_special_characters(criteria, replace=''))
    else:
        string_count_variable = string_count_variable
    
    if criteria=='*':
        DataFrame[string_count_variable] = DataFrame[variable].str.len()
    else:
        DataFrame[string_count_variable] = DataFrame[variable].str.count(criteria) 
            
    if return_variable:
        return DataFrame, string_count_variable
    else:
        return DataFrame
    
def create_string_count_variables(DataFrame, string_counts, return_variables=False):
    string_count_variables = []
    for string_count in string_counts:
        try:
            DataFrame, string_count_variable = count_characters(DataFrame, string_count['variable'], string_count['criteria'], string_count['count_variable'], return_variable=True)
            string_count_variables.append(string_count_variable)            
        except:
            print('ERROR creating string count variable {} counting {} : {}'.format(string_counts['variable'], string_counts['criteria'], traceback.format_exc()))  
        
    if return_variables:
        return DataFrame, string_count_variables
    else:
        return DataFrame        
    
def create_string_similarity_variable(DataFrame, variable1, variable2, string_similarity_variable=None, metric=None, case_sensitive=True, min_length=1, max_length=np.inf, normalize=False, return_variable=False):
    if string_similarity_variable==None:
        string_similarity_variable = '{}SIM{}'.format(variable1,variable2)
    else:
        string_similarity_variable = string_similarity_variable
    
    if metric=='levenshtein':
        DataFrame[string_similarity_variable] = np.vectorize(damerau_levenshtein_distance)(DataFrame[variable1], DataFrame[variable2], case_sensitive, normalize)
    elif metric=='jaccard':
        method='substring'
        DataFrame[string_similarity_variable] = np.vectorize(jaccard_index)(DataFrame[variable1], DataFrame[variable2], method, case_sensitive, min_length, max_length)

    if return_variable:
        return DataFrame, string_similarity_variable
    else:
        return DataFrame

def create_string_similarity_variables(DataFrame, string_similarity, return_variables=False):
    string_similarity_variables = []
    for similarity in string_similarity:
        try:
            try:
                min_length = similarity['parameters']['min_length']
                max_length = similarity['parameters']['max_length']
                normalize = similarity['parameters']['normalize']
            except:
                min_length = 1, 
                max_length = np.inf
                normalize = False
            DataFrame, string_similarity_variable = create_string_similarity_variable(DataFrame, similarity['variable1'], similarity['variable2'], similarity['string_similarity_variable'], similarity['metric'], similarity['case_sensitive'], min_length, max_length, normalize, return_variable=True)
            string_similarity_variables.append(string_similarity_variable)            
        except:
            print('ERROR creating string similarity {} between variables {} and {}: {}'.format(similarity['metric'], similarity['variable1'], similarity['variable2'], traceback.format_exc()))  
        
    if return_variables:
        return DataFrame, string_similarity_variables
    else:
        return DataFrame  
    
def create_target_variable(DataFrame, target, return_variable=False):
    target_variables = []
    
    for tgt in target:
        try:
            target_variable=tgt['target_variable']
            target_condition=tgt['str_condition'] 
            DataFrame = set_binary_target(DataFrame, target_condition=target_condition, target_variable=target_variable)
            target_variables.append(target_variable)
        except:
            print('ERROR creating target variable {} = {} :\n {}'.format(target_variable, target_condition, traceback.format_exc()))  

    if return_variable:
        return DataFrame, target_variables
    else:
        return DataFrame
    
###############################################################################
##[ ENCODER ]##################################################################      
############################################################################### 
def to_one_hot_encode(DataFrame, category_variables=[], binary_variables=[], target_variable='target', target_type='binary'):
    # TO DO: If target type is 'multi' apply one hot encoding to target
    try:
        VariablesDummies = pd.get_dummies(DataFrame[category_variables]).astype('int8')
        dummy_variables = list(VariablesDummies.columns.values)
        DataFrame[dummy_variables] = VariablesDummies
    except:
        print('Category columns {} does not specified nor exists'.format(category_variables))
        
    try:
        DataFrame[binary_variables] = DataFrame[binary_variables].astype('int8')
    except:
        print('Binary columns {} does not specified nor exists'.format(binary_variables))
              
    feature_variables = binary_variables+dummy_variables
    
    return DataFrame, feature_variables, target_variable

###############################################################################
##[ ML MODEL DRIVER ]##########################################################      
###############################################################################     
def load_data_task(load_data_dict, return_name=False):
    """
    Parameters
    ----------
    load_data_dict: dict
    e.g.:   {
        	  "type": "csv",
        	  "location": "local",
        	  "workclass": "Private",
        	  "source": {"path":"C:/Projects/Data/incomedata.csv", "separator":",", "encoding":null},
        	  "auth": None,
        	  "query": None,
        	  "limit": None
            }
    
    Returns
    -------
    DataFrame: pandas.DataFrame
    data_name: str
    """    

    import json
    if type(load_data_dict)==dict:
        pass
    else:
        try:
            load_data_dict = json.loads(load_data_dict) 
        except:
            print('ERROR in loading data:{}\n {}'.format(load_data_dict, traceback.format_exc()))  
    
    data_name = load_data_dict['data_name']
        
    if load_data_dict['type']=='csv':
        DataFrame = read_data_csv(
                file=load_data_dict['source']['path'], 
                separator=load_data_dict['source']['separator'], 
                encoding=load_data_dict['source']['encoding']
                )
    elif load_data_dict['type']=='pickle':
        DataFrame = read_data_pickle(
                file=load_data_dict['source']['path'], 
                compression =load_data_dict['source']['compression']
                )    
    elif load_data_dict['type']=='sql':
        DataFrame = read_data_sql(
                query=load_data_dict['query'], 
                server=load_data_dict['source']['server'], 
                database=load_data_dict['source']['database'],
                auth=load_data_dict['auth']
                )    
    else:
        print("No valid data source provided!")
        DataFrame = pd.DataFrame()	

    # Add ID column
    DataFrame = add_identity_column(DataFrame, id_label='ID', start=1, increment=1)

    # Clean column names
    DataFrame = clean_column_names(DataFrame, replace='')
        
    if return_name:  
        return DataFrame, data_name
    else:
        return DataFrame

    
def create_variable_task(DataFrame, create_variable_task_dict=None, return_extra=False):
    
    import json
    if type(create_variable_task_dict)==dict:
        pass
    else:
        try:
            create_variable_task_dict = json.loads(create_variable_task_dict) 
        except:
            print('ERROR in creating variable:{}\n {}'.format(create_variable_task_dict, traceback.format_exc()))  
    
    if create_variable_task_dict['type']=='target':
        target = create_variable_task_dict['rule_set']
        variable_class = create_variable_task_dict['variable_class']
        include = create_variable_task_dict['include']
        # Apply operations on variables
        DataFrame, output_variables = create_target_variable(DataFrame, target, return_variable=True)        
    elif create_variable_task_dict['type']=='transform':
        operations = create_variable_task_dict['rule_set']
        variable_class = create_variable_task_dict['variable_class']
        include = create_variable_task_dict['include']
        # Apply operations on variables
        DataFrame, output_variables = create_transformed_variables(DataFrame, operations, return_variables=True)
    elif create_variable_task_dict['type']=='conditions':
        conditions = create_variable_task_dict['rule_set']
        variable_class = create_variable_task_dict['variable_class']
        include = create_variable_task_dict['include']
        # Apply operations on variables
        DataFrame, output_variables = create_binary_variables(DataFrame, conditions, return_variables=True) 
    elif create_variable_task_dict['type']=='buckets':
        buckets = create_variable_task_dict['rule_set']
        variable_class = create_variable_task_dict['variable_class']
        include = create_variable_task_dict['include']
        # Create more Catergorical variables
        DataFrame, output_variables = create_categorical_variables(DataFrame, buckets, return_variables=True)
    elif create_variable_task_dict['type']=='category_merges':
        category_merges = create_variable_task_dict['rule_set']
        variable_class = create_variable_task_dict['variable_class']
        include = create_variable_task_dict['include']
        # Merge categorical values
        DataFrame, output_variables = merge_categories(DataFrame, category_merges, return_variables=True)
    elif create_variable_task_dict['type']=='pair_equality':
        variable_pairs = create_variable_task_dict['rule_set']
        variable_class = create_variable_task_dict['variable_class']
        include = create_variable_task_dict['include']
        # Merge categorical values
        DataFrame, output_variables = create_variable_pair_equality(DataFrame, variable_pairs, return_variables=True)
    elif create_variable_task_dict['type']=='string_similarity':
        string_similarity = create_variable_task_dict['rule_set']
        variable_class = create_variable_task_dict['variable_class']
        include = create_variable_task_dict['include']
        # Merge categorical values
        DataFrame, output_variables = create_string_similarity_variables(DataFrame, string_similarity, return_variables=True)
    elif create_variable_task_dict['type']=='string_counts':
        string_counts = create_variable_task_dict['rule_set']
        variable_class = create_variable_task_dict['variable_class']
        include = create_variable_task_dict['include']
        # Merge categorical values
        DataFrame, output_variables = create_string_count_variables(DataFrame, string_counts, return_variables=True)
    elif create_variable_task_dict['type']=='date_differences':
        date_differences = create_variable_task_dict['rule_set']
        variable_class = create_variable_task_dict['variable_class']
        include = create_variable_task_dict['include']
        # Merge categorical values
        DataFrame, output_variables = create_date_difference_variables(DataFrame, date_differences, return_variables=True)
    elif create_variable_task_dict['type']=='numeric_differences':
        numeric_differences = create_variable_task_dict['rule_set']
        variable_class = create_variable_task_dict['variable_class']
        include = create_variable_task_dict['include']
        # Merge categorical values
        DataFrame, output_variables = create_numeric_difference_variables(DataFrame, numeric_differences, return_variables=True)        
    else:
        output_variables= None    
        variable_class = None
        include = False
    
    if return_extra:    
        return DataFrame, output_variables, variable_class, include   
    else:
        return DataFrame, output_variables



def setup_variables_task(DataFrame, variables_setup_dict):
    import re
    import json
    if type(variables_setup_dict)==dict:
        pass
    else:
        try:
            variables_setup_dict = json.loads(variables_setup_dict) 
        except:
            print('ERROR in creating variables:{}\n {}'.format(variables_setup_dict, traceback.format_exc()))  
            
    # Setting = {'model', 'score'}     
    setting = variables_setup_dict['setting']
    
    # verify if variables exists
    category_variables = variables_setup_dict['variables']['category_variables']
    binary_variables = variables_setup_dict['variables']['binary_variables']  
    target_variable = variables_setup_dict['variables']['target_variable'] 
    
    #Create variables sets
    category_variables =  set(category_variables) & set(DataFrame.columns)
    binary_variables  = set(binary_variables) & set(DataFrame.columns)

    # Check if target variable exists (fill the column with None in scoring)
    if not target_variable in DataFrame.columns:
        DataFrame[target_variable]=None    
    
    # Run variable creation task list
    for task_type in variables_setup_dict.keys():
        task_type = re.sub('[\W\d]', '', task_type)         
        if task_type in ['target', 'transforms', 'conditions', 
                 'buckets', 'category_merges', 'pair_equality', 'string_entity', 
                 'string_similarity', 'string_counts']:
            #print(task_type)
            DataFrame, variable_, variable_class_, include_ = create_variable_task(DataFrame, create_variable_task_dict= variables_setup_dict[task_type], return_extra=True)        

            if variable_class_=='bin' and include_:
                binary_variables.update(variable_)
            elif variable_class_=='cat' and include_:
                category_variables.update(variable_)

    #Finalize variables lists
    category_variables=list(category_variables)
    binary_variables=list(binary_variables)
    target_variable = target_variable
    
    return DataFrame, category_variables, binary_variables, target_variable
