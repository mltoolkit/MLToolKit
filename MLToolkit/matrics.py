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

###############################################################################
# PLOT ROBUSTNESS TABLE 
def plot_model_results(ResultsTable, x_column, y_column, size_column, color_column, color_scale=100, size_scale=2000):
    from matplotlib import colors #For custom color maps
    ResultsTable=ResultsTable.astype(dtype='float32')
    bounds = np.array([0.0, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 25.0, 50.0])
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    ResultsTable.sort_values(by=size_column, ascending=False, inplace=True)
    plt.figure()
    plt.title('Model Charateristics \n ResponseFraction ~ Marker size ')
    plt.scatter(ResultsTable[x_column], ResultsTable[y_column], c=ResultsTable[color_column].values*color_scale, s=ResultsTable[size_column]*size_scale, cmap='nipy_spectral', norm=norm, marker='s')
    plt.plot([0, 100], [0, 100], 'k:')
    plt.xlabel(x_column + ' (Predicted)')
    plt.ylabel(y_column+ ' (Actual)')
    cbar = plt.colorbar()
    cbar.set_label(color_column)
    
    plt.xlim(0, max(ResultsTable[x_column])*1.05)
    plt.ylim(0, max(ResultsTable[y_column])*1.05)
    
###############################################################################
# ROBUSTNESS TABLE                 
def robustness_table(ResultsSet, class_variable='Response', score_variable='Probability',  score_label='Score', show_plot=False):  

    RobustnessTable = ResultsSet.groupby(by=[score_label]).agg({score_variable:[min,max,'mean','count'], class_variable:sum})        
    
    RobustnessTable.columns=np.array(['min{}'.format(score_variable), 'max{}'.format(score_variable), 'mean{}'.format(score_variable), 'BucketCount', 'ResponseCount'])
    
    mean_probability = np.mean(ResultsSet[score_variable])
    min_probability = np.min(ResultsSet[score_variable])
    max_probability = np.max(ResultsSet[score_variable])
    
    total_count = np.sum(RobustnessTable['BucketCount'])
    total_response_count = np.sum(RobustnessTable['ResponseCount'])
        
    RobustnessTable['BucketFraction'] = RobustnessTable['BucketCount']/total_count
    RobustnessTable['ResponseFraction'] =RobustnessTable['ResponseCount']/total_response_count
    RobustnessTable['BucketPrecision'] = RobustnessTable['ResponseCount']/RobustnessTable['BucketCount'] 
    RobustnessTable['CumulativeBucketFraction'] = RobustnessTable['BucketFraction'][::-1].cumsum()
    RobustnessTable['CumulativeResponseFraction'] = RobustnessTable['ResponseFraction'][::-1].cumsum()
    RobustnessTable['CumulativePrecision'] = RobustnessTable['ResponseCount'][::-1].cumsum()/RobustnessTable['BucketCount'][::-1].cumsum()
#    RobustnessTable['CumulativeBucketCount'] = RobustnessTable['BucketCount'][::-1].cumsum()
#    RobustnessTable['CumulativeResponseCount'] = RobustnessTable['ResponseCount'][::-1].cumsum()
    
    total_bucket_fraction = total_count/total_count
    total_response_fraction = total_response_count/total_response_count
    mean_precision= np.sum(RobustnessTable['BucketPrecision']*RobustnessTable['BucketCount'])/total_count
    
    SummaryRow = pd.DataFrame(data=[[min_probability, max_probability, mean_probability, total_count, total_response_count, total_bucket_fraction, total_response_fraction, mean_precision, total_bucket_fraction, total_response_fraction, mean_precision]], columns=RobustnessTable.columns)
    SummaryRow.index = np.array(['DataSet'])
    SummaryRow.index.name =RobustnessTable.index.name

    #Append Summary to the table
    RobustnessTable = RobustnessTable.append(SummaryRow, ignore_index=False)
    
    if show_plot==True:
        plot_model_results(RobustnessTable[RobustnessTable.index!='DataSet'], x_column='mean{}'.format(score_variable), y_column='BucketPrecision', size_column='ResponseFraction', color_column='BucketFraction', color_scale=100, size_scale=2000)
                
    return RobustnessTable

###############################################################################
# COMPUTE MODEL PERFORMANCE EVALUATION MATRICS                  
def model_performance_matrics(ResultsSet, class_variable='Actual', score_variable='Probability', quantile_label='Quantile',  quantiles=1000, show_plot=False):
    from sklearn import metrics #roc_curve, auc, precision_recall_curve,balanced_accuracy_score

    # Create quantiles
    ResultsSet[quantile_label] = pd.qcut(x=ResultsSet[score_variable], q=quantiles, labels = False, duplicates='drop')
    ResultsSet[quantile_label] = ResultsSet[quantile_label] + 1
    
    RobustnessTable = robustness_table(ResultsSet, class_variable=class_variable, score_variable=score_variable, score_label=quantile_label, show_plot=show_plot)    
    #RobustnessTable[:-1].plot(x='maxProbability', y=['CumulativePrecision', 'CumulativeBucketFraction', 'CumulativeResponseFraction'], xlim=[0.0, 1.0], ylim=[0.0, 1.05])
    #RobustnessTable[:-1].plot(x='meanProbability', y=['BucketPrecision'], xlim=[0.0, 1.0], ylim=[0.0, 1.05])
    
    ROCCurve = {}  
    ROCCurve['FPR'], ROCCurve['TPR'], ROCCurve['Threshold'] = metrics.roc_curve(ResultsSet[class_variable].values, ResultsSet[score_variable].values)
    AUC = metrics.auc( ROCCurve['FPR'], ROCCurve['TPR'])
    ROCCurve = pd.DataFrame(data=ROCCurve)
    
    # undersample curve     
    if len(ROCCurve.index)>10000:
        a = ROCCurve[:1]
        b = ROCCurve[1:-1].sample(10000-2).sort_values(by='Threshold', ascending=False)
        c = ROCCurve[-1:] 
        ROCCurve = a.append(b).append(c).reset_index(drop=True)

    PrecisionRecallCurve = {}  
    PrecisionRecallCurve['Precision'], PrecisionRecallCurve['Recall'], PrecisionRecallCurve['Threshold'] = metrics.precision_recall_curve(ResultsSet[class_variable].values, ResultsSet[score_variable].values)
    PrecisionRecallCurve['Threshold']=np.insert(PrecisionRecallCurve['Threshold'], 0,0)    
    PrecisionRecallCurve = pd.DataFrame(data=PrecisionRecallCurve)
    
    # undersample curve
    if len(PrecisionRecallCurve.index)>10000:
        a = PrecisionRecallCurve[:1]
        b = PrecisionRecallCurve[1:-1].sample(10000-2).sort_values(by='Threshold', ascending=True)
        c = PrecisionRecallCurve[-1:]
        PrecisionRecallCurve = a.append(b).append(c).reset_index(drop=True)
    
    return RobustnessTable, ROCCurve, PrecisionRecallCurve, AUC

###############################################################################
#  PLOT EVALUATION MATRICS                  
def plot_eval_matrics(RobustnessTable, ROCCurve, PrecisionRecallCurve, AUC, score_variable, figure=1, description=''):
    import matplotlib.pyplot as plt
    from matplotlib import colors #For custom color maps
       
    plt.figure(figure, figsize=(8, 6), dpi=80)
    
    ax0 = plt.subplot(231) 
    plt.plot(ROCCurve.FPR.values, ROCCurve.TPR.values, linestyle='-', label='{} (area = {:.2f}) '.format(description, AUC))
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()
    
    ax1 = plt.subplot(234) 
    plt.plot(PrecisionRecallCurve['Recall'].values, PrecisionRecallCurve['Precision'].values, linestyle='-', label='{}'.format(description))
    plt.legend()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    
    ax2 = plt.subplot(232, sharey=ax1)
    plt.plot(PrecisionRecallCurve['Threshold'].values, PrecisionRecallCurve['Precision'].values, linestyle='-', label='{}'.format(description))
    #plt.plot(thresholds, recall[:-1], 'o-', label='{} recall'.format(description))
    plt.legend()
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.title('Precision vs. Threshold Curve')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    ax3 = plt.subplot(233, sharex=ax2)
    RobustnessTable=RobustnessTable.copy()[:-1]
    #plt.plot(thresholds, precision[:-1], 'o-', label='{} precision'.format(description))
    plt.plot(RobustnessTable['max{}'.format(score_variable)], RobustnessTable['CumulativeBucketFraction'], linestyle='-', label='{}'.format(description)) #, marker='o'
    plt.legend()
    plt.xlabel('Threshold')
    plt.ylabel('Bucket Fraction')
    plt.title('Bucket Fraction vs. Threshold Curve')
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
 
    ax5 = plt.subplot(236)
    x_column = 'mean{}'.format(score_variable)
    y_column = 'BucketPrecision'
    color_column = 'BucketFraction'
    size_column = 'ResponseFraction'     
#    ColorScale=100
    SizeScale=2000
    size_offset=1      		

#    bounds = np.array([0.0, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 25.0, 50.0])
#    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    RobustnessTable.sort_values(by=size_column, ascending=False, inplace=True)
#    plt.scatter(RobustnessTable[x_column], RobustnessTable[y_column], c=RobustnessTable[color_column].values*ColorScale, s=RobustnessTable[size_column].values*SizeScale, cmap='nipy_spectral', norm=norm, marker='s')
    plt.scatter(RobustnessTable[x_column], RobustnessTable[y_column], s=size_offset+RobustnessTable[size_column].values*size_scale, label='{}'.format(description))
    plt.legend()
    plt.xlabel(x_column)
    plt.ylabel(y_column)
#    plt.title('Model Charateristics \n ResponseFraction ~ Marker size')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
#    cbar = plt.colorbar()
#    cbar.set_label(color_column)  
    
    plt.subplots_adjust(hspace=0.4)
#    plt.show()

def confusion_matrix(actual_variable, predcted_variable, labels=None, sample_weight=None, totals=False):
    from sklearn.metrics import confusion_matrix
    CF = confusion_matrix(actual_variable, predcted_variable, labels=labels)
    CF = pd.DataFrame(data=CF)
    
    if labels==None:
        labels=actual_variable.unique()
        
    if totals==True:
        CF[2] = CF.sum(axis=1)
        CF.loc[2] = CF.sum(axis=0)  
        totals_label= [['Total'], ['']]
    else:
        totals_label= [[],[]]

    col_names = [['Predicted']*len(labels)+totals_label[0], labels+totals_label[1]]
    row_names = [['Actual']*len(labels)+totals_label[0], labels+totals_label[1]]
    columns = pd.MultiIndex.from_arrays(col_names)
    index = pd.MultiIndex.from_arrays(row_names)    
    CF.index = index
    CF.columns=columns
    
    return CF
	
def confusion_matrix_to_row(CF, ModelID='M'):     
    if CF.columns.values[-1][0]=='Total':        
        labels= CF.columns.levels[1].values[:-1]
        CF= CF.values[:-1,:-1]
    else:
        labels= CF.columns.levels[1].values
        CF= CF.values
        
    if len(CF)==2:
        # Compute confusion matrix
        TN = CF[0,0]
        FP = CF[0,1]
        FN = CF[1,0]
        TP = CF[1,1]
        TOTAL = TN+FP+FN+TP   

        CFTERMS = pd.DataFrame(data=[[TN, FP, FN, TP, TOTAL]], columns=['TN', 'FP', 'FN', 'TP', 'TOTAL'])
        CFTERMS.insert(0, 'ModelID', ModelID)

        CFTERMS['P1'] = FP+TP
        CFTERMS['P0'] = TN+FN
        CFTERMS['A1'] = TP+FN
        CFTERMS['A0'] = FP+TN

        CFTERMS['TPR'] = TP/(TP+FN) #TP/A1
        CFTERMS['TNR'] = TN/(FP+TN) #TN/A0
        CFTERMS['FPR'] = FP/(FP+TN) #FP/A0    
        CFTERMS['FNR'] = FN/(FN+TP) #FN/A1

        CFTERMS['PPV'] = TP/(TP+FP) #TP/P1W
        CFTERMS['ACC'] = (TP+TN)/TOTAL
        CFTERMS['F1'] = 2*TP/(2*TP+FP+FN)
        
    else:  
        TOTAL = np.sum(CF)
        TPS = np.diagonal(CF)
        SUCCESS = np.sum(TPS) #np.trace(CF)
        
        ASUM = np.sum(CF,axis=1) #Sum of Actual Conditions
        PSUM = np.sum(CF,axis=0) #Sum of Predicted conditions
        PPV = TPS/PSUM
        TPR = TPS/ASUM
               
        CFTERMS = pd.DataFrame(data=[[ModelID, TOTAL, SUCCESS]], columns=['ModelID', 'TOTAL', 'SUCCESS'])
        CFTERMS['ACC'] = SUCCESS/TOTAL  
        for i in range(len(labels)):
            label = str(labels[i])
            CFTERMS[label+'_PPV']=PPV[i]
            CFTERMS[label+'_TPR']=TPR[i]
            
    return CFTERMS
	
