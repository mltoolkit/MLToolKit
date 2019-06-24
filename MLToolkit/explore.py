# -*- coding: utf-8 -*-
# MLToolkit (mltoolkit)

__docformat__ = 'restructuredtext'
__name__="MLToolkit"
__version__="0.1.2"
__author__="Sumudu Tennakoon"
__create_date__="Sun Jul 01 2018"
__last_update__="Sun Jun 22 2018"
__license__="""
Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
__doc__="""
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

Links
-----
Website: http://sumudu.tennakoon.net/projects/MLToolkit
Github: https://github.com/sptennak/MLToolkit
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

def histogram(DataFrame, column, n_bins=10, bin_range=None, orientation='vertical'):       
    counts, edge_labels, bars = plt.hist(DataFrame[column], bins=n_bins, range=bin_range, orientation=orientation, density=False)
    plt.xlabel(column)
    plt.ylabel('bins')
    plt.grid(True)
    
    try:
        n = len(n_bins)
    except:
        n = n_bins
    
    bin_labels=[]    
    for i in range(n):
        l_bracket = '['
        r_bracket = ']'
        bin_labels.append('{}_{}{:g},{:g}{}'.format(i+1, l_bracket, edge_labels[i], edge_labels[i+1], r_bracket))
        
    Table = pd.DataFrame(data={'counts':counts}, dtype='int', index=bin_labels)
    Table.index.name = column
    
    TotalRow = pd.DataFrame(data=[Table['counts'].sum()], columns=['counts'], index=np.array(['TOTAL']))
    TotalRow.index.name = Table.index.name
    Table = Table.append(TotalRow, ignore_index=False)
    
    return Table

def category_lists(DataFrame, categorical_variables):
    out = ''
    for variable in categorical_variables:
        out = out + '{}:{}\n\n'.format(variable, tuple(DataFrame[variable].unique()))
    return out
	
def variable_frequency(DataFrame, variable):   
    x = variable
    FrequencyTable = DataFrame.groupby(by=x)[[x]].count().astype('int')
    FrequencyTable['CountsFraction%'] = FrequencyTable[x]/FrequencyTable[x].sum() * 100
    return FrequencyTable

def variable_response(DataFrame, variable, class_variable):   
    X = variable
    y = class_variable
    ResponseTable = DataFrame.groupby(by=X)[[X,y]].agg({X:'count',y:'sum'}).astype('int')
    ResponseTable['CountsFraction%'] = ResponseTable[X]/ResponseTable[X].sum() * 100
    ResponseTable['ResponseFraction%'] = ResponseTable[y]/ResponseTable[y].sum() * 100
    ResponseTable['ResponseRate%'] = ResponseTable[y]/ResponseTable[X] * 100
    ResponseTable.index = ResponseTable.index.astype(str)

    total_row = [ResponseTable[X].sum(),
                ResponseTable[y].sum(),
                ResponseTable['CountsFraction%'].sum(),
                ResponseTable['ResponseFraction%'].sum(),
                None]
    TotalRow = pd.DataFrame(data=[total_row], columns=ResponseTable.columns, index=np.array(['TOTAL']))
    TotalRow.index.name = ResponseTable.index.name
    ResponseTable = ResponseTable.append(TotalRow, ignore_index=False)
    
    ResponseTable.rename(index=str, columns={X:'Counts'}, inplace=True)
    return ResponseTable

def plot_variable_response(DataFrame, variable, class_variable):
    X = variable
    y = class_variable   
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
    plt.show()
        
def plot_variable_responses(DataFrame, variables, class_variable):
    y = class_variable
    for X in variables:
        plot_variable_response(DataFrame, variable=X, class_variable=y)	

def correlation_matrix_to_list(correlation, ascending=False):
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
    
    return correlation_list.sort_values(by=['|Correlation|'], ascending=ascending) 

def correlation_matrix(DataFrame, feature_variables, method='pearson', return_type='matrix', show_plot=False):
    '''
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html
    method: {'pearson', 'kendall', 'spearman'} 
    return_type: {'matrix', 'list'}
    '''
    correlation = DataFrame[feature_variables].corr()
    
    if show_plot:
        plt.matshow(correlation)
        plt.show()
        
    if return_type=='list':
        correlation=correlation_matrix_to_list(correlation, ascending=False)
        
    return correlation             
	
def univariate_stats(DataFrame, feature_variables, class_variable):
    from sklearn.metrics import confusion_matrix
    Univariate = []
    for i in range(len(feature_variables)):
        univariate_columns = ['TN', 'FP', 'FN', 'TP']
        Univariate.append(confusion_matrix(DataFrame[class_variable],DataFrame[feature_variables[i]], labels=[0,1]).ravel())
    
    Univariate = pd.DataFrame(data=Univariate, index=feature_variables, columns=univariate_columns, dtype='int')
    Univariate['F1'] = 2 * Univariate['TP'] / (2*Univariate['TP'] + Univariate['FP'] + Univariate['FN'])
    Univariate['VariablePositive'] = Univariate['FP'] + Univariate['TP']
    Univariate['TrueResponse'] = Univariate['FN'] + Univariate['TP']
    Univariate['HitRate'] = Univariate['TP']  / (Univariate['TP'] + Univariate['FN']) #Recall
    Univariate['FalsePositiveRate'] = Univariate['FP']  / (Univariate['FP'] + Univariate['TN']) #FPR
    
    return Univariate