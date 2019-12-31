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
Github: https://github.com/mltoolkit/mltk

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
		
def data_description(DataFrame, include='all'):
    DataStats = DataFrame.describe(percentiles=[.1, .25, .5, .75, .9], include=include).transpose()
    DataStats = DataStats.where((pd.notnull(DataStats)), None)
    DataTypes = pd.DataFrame(data=DataFrame.dtypes, columns=['dtypes'])
    DataStats=DataStats.merge(DataTypes, left_index=True, right_index=True, how='left')
    return DataStats
	
def histogram(DataFrame, variable, n_bins=10, bin_range=None, orientation='vertical', density=False, show_plot=False):    
    """
    Parameters
    ----------
    DataFrame : pandas.DataFrame
        DataFrame
    variable : str
        Variable to calculate histogram.    
    n_bins : int, or list(numeric)
        Number of bins or list of bin edges.
    bin_range : 
    orientation : {'json', 'dict'}, default 'json'
    
    Returns
    -------
    out : JSON str or dict
    
    """
    counts, edge_labels = np.histogram(DataFrame[variable], bins=n_bins, range=bin_range, density=False)
    
    try:
        n = len(n_bins)-1
    except:
        n = n_bins
    
    bin_labels=[]    
    for i in range(n):
        l_bracket = '['
        r_bracket = ']'
        bin_labels.append('{}_{}{:g},{:g}{}'.format(i+1, l_bracket, edge_labels[i], edge_labels[i+1], r_bracket))
    
    Table = pd.DataFrame(data={'counts':counts}, dtype='int', index=bin_labels)
    Table.index.name = variable
    
    if density:
        density_, edge_labels_ = np.histogram(DataFrame[variable], bins=n_bins, range=bin_range, density=True)
        Table['density'] = density_
        
    TotalRow = pd.DataFrame(data=[Table['counts'].sum()], columns=['counts'], index=np.array(['TOTAL']))
    TotalRow.index.name = Table.index.name
    Table = Table.append(TotalRow, ignore_index=False)
    
    if show_plot:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        DataFrame[variable].plot(ax=ax1, kind='hist', alpha=0.7, edgecolor='black', range=bin_range, bins=n_bins, grid=True)
        ax_limt = ax1.axis()
        if density and orientation=='vertical':
            DataFrame[variable].plot(ax=ax2, kind='kde', style='r-')
        else:
            print('KDE not supported with horizontal option')
        ax1.set_xlabel(variable)

        ax1.set_xlim((ax_limt[0], ax_limt[1]))
        ax2.set_xlim((ax_limt[0], ax_limt[1]))
        
        plt.show()
    
    return Table

def category_lists(DataFrame, categorical_variables, threshold=50, return_type='json'):
    """
    Parameters
    ----------
    DataFrame : pandas.DataFrame
        DataFrame
    categorical_variables : list(str)
        Variable to examine the freqeuncy.    
    threshold : int, default 50
        Maximum number of categories expected
    return_type : {'json', 'dict'}, default 'json'
    
    Returns
    -------
    out : JSON str or dict
    """
    out = dict()
    for variable in categorical_variables:
        categories = DataFrame[variable].unique()
        if len(categories)>threshold:
            out[variable] = []
            print('Numebr of ategories > {}'.format(threshold))
        else:
            out[variable] = list(categories)
    
    if return_type=='json':
        import json 
        out = json.dumps(out, indent = 4)
        
    return out
	
def variable_frequency(DataFrame, variable, sorted=False, ascending=False, show_plot=False): 
    """
    Parameters
    ----------
    DataFrame : pandas.DataFrame
        DataFrame
    variable : str
        Variable to examine the freqeuncy.    
    sorted : bool, default False
    ascending : bool, default False    
    show_plot : bool, default False
        plot results if True

    Returns
    -------
    FrequencyTable : pandas.DataFrame
    """
    x = variable
    FrequencyTable = DataFrame.groupby(by=x)[[x]].count().astype('int')
    FrequencyTable['CountsFraction%'] = FrequencyTable[x]/FrequencyTable[x].sum() * 100
    FrequencyTable.rename(index=str, columns={x:'Counts'}, inplace=True)
    if sorted:
        FrequencyTable.sort_values(by=['Counts'], ascending=ascending, inplace =True)
        
    total_row = [FrequencyTable['Counts'].sum(), FrequencyTable['CountsFraction%'].sum()]
    TotalRow = pd.DataFrame(data=[total_row], columns=FrequencyTable.columns, index=np.array(['TOTAL']))
    TotalRow.index.name = FrequencyTable.index.name
    FrequencyTable = FrequencyTable.append(TotalRow, ignore_index=False)    
        
    
    if show_plot:
        FrequencyTable.loc[FrequencyTable.index!='TOTAL'][['Counts']].plot(kind='bar')
        
    return FrequencyTable

def variable_response(DataFrame, variable, target_variable, measurement_variable=None, condition=None, sort_by=None, ascending=False, show_plot=False):  
    """
    Parameters
    ----------
    DataFrame : pandas.DataFrame
        DataFrame
    variable : str
        Variable to examine the freqeuncy.
    target_variable : str
        Target variable
    condition, str, default None 
        Filtering condition
    sorted_by : {'count','response','rate'}, default None
    ascending : bool, default False
    show_plot : bool, default False
        plot results if True
        
    Returns
    -------
    ResponseTable : pandas.DataFram
    """
    count_flag = '__count__'
    DataFrame[count_flag] = 1
    
    X = variable
    y = target_variable
    
    if measurement_variable!=None and DataFrame[measurement_variable].dtype in (
            'int_', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 
                'uint32', 'uint64', 'float_', 'float16', 'float32', 'float64'):
        y0 = measurement_variable
    else:
        y0 = count_flag
    
    if condition!=None:
        try:
            DataFrame = DataFrame.query(condition)
        except:
            print('Filtering with {} failed'.format(condition))
    
    ResponseTable = DataFrame.groupby(by=X)[[y0,y]].agg({y0:'sum',y:'sum'}).astype('int64')
    DataFrame.drop(columns=[count_flag])
    
    ResponseTable['CountsFraction%'] = ResponseTable[y0]/ResponseTable[y0].sum() * 100
    ResponseTable['ResponseFraction%'] = ResponseTable[y]/ResponseTable[y].sum() * 100
    ResponseTable['ResponseRate%'] = ResponseTable[y]/ResponseTable[y0] * 100
    ResponseTable.index = ResponseTable.index.astype(str)
    
#    # Following two lines is to void index name conflict with the column # [1] Imrove this
#    index_name = ResponseTable.index.name # [1] Imrove this
#    ResponseTable.index.name = 'index' # [1] Imrove this
    
    if sort_by=='count':        
        ResponseTable.sort_values(by=[y0, y], ascending=ascending, inplace=True, na_position='last')
    elif sort_by=='response':        
        ResponseTable.sort_values(by=[y, 'ResponseRate%'], ascending=ascending, inplace=True, na_position='last')
    elif sort_by=='rate':        
        ResponseTable.sort_values(by=['ResponseRate%', y], ascending=ascending, inplace=True, na_position='last')    
    
    total_row = [ResponseTable[y0].sum(),
                ResponseTable[y].sum(),
                ResponseTable['CountsFraction%'].sum(),
                ResponseTable['ResponseFraction%'].sum(),
                ResponseTable[y].sum()/ResponseTable[y0].sum() * 100]
    TotalRow = pd.DataFrame(data=[total_row], columns=ResponseTable.columns, index=np.array(['TOTAL']))
    TotalRow.index.name = ResponseTable.index.name
    ResponseTable = ResponseTable.append(TotalRow, ignore_index=False)
    
#    ResponseTable.index.name = index_name # [1] Imrove this
    ResponseTable.rename(index=str, columns={y0:'Counts'}, inplace=True)
    
    if show_plot:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ResponseTable.loc[ResponseTable.index!='TOTAL'][['Counts']].plot(ax=ax1, kind='bar')
        ResponseTable.loc[ResponseTable.index!='TOTAL'][['ResponseRate%']].plot(ax=ax2, kind='line', style='r-o')
        ResponseTable.loc[ResponseTable.index!='TOTAL'][['ResponseFraction%']].plot(ax=ax2, kind='line', style='k-o')
        ax1.set_ylabel('Counts')
        ax2.set_ylabel('ResponseRate, ResponseFraction %')
        plt.title(y)
        line1, label1 = ax1.get_legend_handles_labels()
        line2, label2 = ax2.get_legend_handles_labels()
        ax1.legend().set_visible(False)
        ax2.legend().set_visible(False)
        plt.legend(line1+line2, label1+label2)
        plt.show() ###    
    return ResponseTable
	
def slice_variable_response(DataFrame, variable, condition):
    """
    Parameters
    ----------
    DataFrame : pandas.DataFrame
        DataFrame
    variable : str
        Variable to examine the freqeuncy.
    condition, str, default None 
        Filtering condition
        
    Returns
    -------
    None
    """
    for X in sorted(DataFrame[variable].unique()):
        print('\n\n{}\n'.format(X))
        print("(ExcludeInModel==0) & ({}=='{}')".format(variable,X))
        try:
            print(variable_response(DataFrame, variable='DupePrevScore', target_variable='DupeToCapture', condition="({}) & ({}=='{}')".format(condition, variable,X), show_plot=True))
        except:
            print('No resuts found !')

def variable_responses(DataFrame, variables, target_variable, show_output=True, show_plot=False):
    """
    Parameters
    ----------
    DataFrame : pandas.DataFrame
        DataFrame
    variable : str
        Variable to examine the freqeuncy.
    target_variable : str
        Target variable
    show_output, bool, default True 
        Print output table
    show_plot : bool, default False
        plot results if True
        
    Returns
    -------
    None
    """
    y = target_variable
    for X in variables:
        output = variable_response(DataFrame, variable=X, target_variable=y, show_plot=show_plot, sort_by=None)
        if show_output:
            print(output)
            
def plot_variable_response(DataFrame, variable, target_variable):
    X = variable
    y = target_variable   
    ResponseTable = variable_response(DataFrame, X, y)
    print(ResponseTable)
    ResponseTable = ResponseTable.loc[ResponseTable.index!='TOTAL']
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ResponseTable[['Counts']].plot(ax=ax1, kind='bar')
    ResponseTable[['ResponseRate%']].plot(ax=ax2, kind='line', style='r-o')
    ResponseTable[['ResponseFraction%']].plot(ax=ax2, kind='line', style='k-o')
    ax1.set_ylabel('Counts')
    ax2.set_ylabel('ResponseRate, ResponseFraction %')
    plt.title(y)
    line1, label1 = ax1.get_legend_handles_labels()
    line2, label2 = ax2.get_legend_handles_labels()
    ax1.legend().set_visible(False)
    ax2.legend().set_visible(False)
    plt.legend(line1+line2, label1+label2)
        
def plot_variable_responses(DataFrame, variables, target_variable):
    y = target_variable
    for X in variables:
        plot_variable_response(DataFrame, variable=X, target_variable=y)	

def correlation_matrix_to_list(correlation, target_variable=None, ascending=False):
    """
    Parameters
    ----------
    correlation : pandas.DataFrame
        Correlation matrix
    target_variable : str
        Target variable
    ascending : bool, default False
        Sort condition
        
    Returns
    -------
    correlation_list : pandas.DataFrame
    """
    variables=correlation.columns.values
    n = len(variables)
    correlation_list = []
    for i in range(n):
        for j in range(i,n): 
            if i!=j:
                index = variables[i]
                column = variables[j]
                corr = np.round(correlation.at[index, column], 5)
                correlation_list.append([index, column, corr, abs(corr)])
    correlation_list = pd.DataFrame(data=correlation_list, columns=['Variable1', 'Variable2', 'Correlation', '|Correlation|'])
    
    try:
        correlation_with_response = correlation_list.loc[correlation_list['Variable2']==target_variable][['Variable1', 'Correlation']]
        correlation_with_response.columns = ['Variable_', 'corrTargetVariable']
        correlation_list = correlation_list.loc[correlation_list['Variable2']!=target_variable]
        correlation_list = correlation_list.merge(correlation_with_response[['Variable_', 'corrTargetVariable']], left_on='Variable1', right_on='Variable_', suffixes=('1', '2'), how='left')
        correlation_list = correlation_list.merge(correlation_with_response[['Variable_', 'corrTargetVariable']], left_on='Variable2', right_on='Variable_', suffixes=('1', '2'), how='left')
        correlation_list = correlation_list[['Variable1', 'Variable2', 'Correlation', '|Correlation|', 'corrTargetVariable1', 'corrTargetVariable2']]
    except:
        pass
    
    return correlation_list.sort_values(by=['|Correlation|'], ascending=ascending) 

def correlation_matrix(DataFrame, variables, target_variable=None, method='pearson', return_type='matrix', show_plot=False):
    """
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html
    
    Parameters
    ----------
    DataFrame : pandas.DataFrame
        DataFrame
    variables : list(str)
        List of variables
    target_variable : str
        Target variable
    method: {'pearson', 'kendall', 'spearman'} 
    return_type: {'matrix', 'list'}
    show_plot : bool, default False
        plot results if True
        
    Returns
    -------
    correlation : pandas.DataFrame
    """

    correlation = DataFrame[variables].corr()
    
    if show_plot:
        f = plt.figure(figsize=(8, 6))
        plt.matshow(correlation, fignum=f.number)
        #plt.xticks(range(DataFrame.shape[1]), DataFrame.columns, fontsize=12, rotation=90)
        #plt.yticks(range(DataFrame.shape[1]), DataFrame.columns, fontsize=12)        
        cb = plt.colorbar()
        #cb.ax.tick_params(labelsize=14)
        plt.title('Correlation Matrix', fontsize=16)
        plt.show()
        
    if return_type=='list':
        correlation=correlation_matrix_to_list(correlation, target_variable=target_variable, ascending=False)
        
    return correlation               
	
def univariate_stats(DataFrame, feature_variables, target_variable):
    """
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html
    
    Parameters
    ----------
    DataFrame : pandas.DataFrame
        DataFrame
    feature_variables : list(str)
        List of variables
    target_variable : str
        Target variable
        
    Returns
    -------
    Univariate : pandas.DataFrame
    """
    from sklearn.metrics import confusion_matrix
    Univariate = []
    for i in range(len(feature_variables)):
        univariate_columns = ['TN', 'FP', 'FN', 'TP']
        Univariate.append(confusion_matrix(DataFrame[target_variable],DataFrame[feature_variables[i]], labels=[0,1]).ravel())
    
    Univariate = pd.DataFrame(data=Univariate, index=feature_variables, columns=univariate_columns, dtype='int')
    Univariate['F1'] = 2 * Univariate['TP'] / (2*Univariate['TP'] + Univariate['FP'] + Univariate['FN'])
    Univariate['VariablePositive'] = Univariate['FP'] + Univariate['TP']
    Univariate['TrueResponse'] = Univariate['FN'] + Univariate['TP']
    Univariate['HitRate'] = Univariate['TP']  / (Univariate['TP'] + Univariate['FN']) #Recall
    Univariate['FalsePositiveRate'] = Univariate['FP']  / (Univariate['FP'] + Univariate['TN']) #FPR
    
    return Univariate