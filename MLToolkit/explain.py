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
import shap
import lime
import traceback
import time 
import matplotlib.pyplot as plt

class MLExplainer():
    def __init__(self, model_id=None, explainer_object=None, explainer_config=None, model_parameters=None):
        self.model_id = model_id
        self.explainer_object = explainer_object
        self.explainer_config = explainer_config
        self.model_parameters = model_parameters

    def get_explainer_object(self):
        return self.explainer_object

    def get_explainer_config(self):
        return self.explainer_config

    def get_model_parameters(self):
        return self.model_parameters

    def get_model_parameter(self, param):
        return self.model_parameters[param]
    
    def get_explainer_config_item(self, item):
        return self.explainer_config[item]
    
    def get_explainer_method(self):
        return self.explainer_config['Method'] 
    
    def get_model_variables(self):
        return self.model_parameters['ModelVariables'] 
        
    def get_base_value(self, class_number=None):
        try:
            if class_number == None:
                class_number = self.explainer_config['ClassNumber']
            else:
                class_number = 0
            return self.explainer_object.expected_value[class_number]
        except:
            print('Base Value calculating Error !\n{}'.format(traceback.format_exc()))  
            return None

def load_object(file_name):
    try:
        import pickle
        pickle_in = open(file_name,'rb')
        print('Loading explainer from file {}'.format(file_name))
        object = pickle.load(pickle_in)
        pickle_in.close()
        return object
    except:
        print('ERROR in loading explainer: {}'.format(traceback.format_exc()))
        return None
    
def save_object(object_to_save, file_name):
    try:
        import pickle
        pickle_out = open(file_name,'wb')
        print('Saving explainer to file {}'.format(file_name))
        pickle.dump(object_to_save, pickle_out)
        pickle_out.close()
    except:
        print('ERROR in saving explainer: {}'.format(traceback.format_exc()))

def save_explainer(Explainer, file_path):
    """
    Parameters
    ----------
    Explainer : mltk.MLExplainer
    file_path : str, dafault ''
        Path to file including the file name.    
    
    Returns
    -------
    None
    """      
    save_object(Explainer, file_path)

def load_explainer(file_path):
    """
    Parameters
    ----------
    file_path : str, dafault ''
        Path to file including the file name.    
    
    Returns
    -------
    Explainer : mltk.MLExplainer
    """
    Explainer = load_object(file_path)
    return Explainer
        
def build_explainer(Model, explainer_config={'Method':'shap'}, train_variable_values=None):
    """    
    Parameters
    ----------    
    Model : mltk.MLModel   
    explainer_config : dict or json, optional  
    train_variable_values : pd.DataFrame (optional for Linear Explainer)
    
    Returns
    -------  
    Explainer : mltk.MLExplainer
    """
    try:
        model_id = Model.get_model_id()
        ml_algorithm = Model.get_model_algorithm()
        model = Model.get_model_object()
        model_type = Model.get_model_type()
        model_variables = Model.get_model_variables()
        predicted_value_column = Model.get_score_variable()
    except:
        print('Error in processing Model\n{}'.format(traceback.format_exc()))
        return None
    
    method = explainer_config['Method']
    
    if 'IdColumns' not in explainer_config:
        explainer_config['IdColumns'] = [] 

    if 'ClassNumber' not in explainer_config:
        explainer_config['ClassNumber'] = 0
    
    model_parameters = {
            'ModelType':model_type,
            'ModelVariables':model_variables,
            'PredictedValueColumn':predicted_value_column
            }
    
    try:
        if method=='shap':
            if ml_algorithm in ['RF']:
                explainer_object = shap.TreeExplainer(model)
            elif ml_algorithm in ['LGR']:
                explainer_object = shap.LinearExplainer(model, train_variable_values[model_variables].values, feature_dependence="independent")
            elif ml_algorithm in ['NN']:
                explainer_object = shap.DeepExplainer(model, train_variable_values[model_variables].values)       
            else:
                explainer_object = None
                print('Model not supported')
        elif method=='lime':
            explainer_object = None
            # Not implemented
        print('Explainer created ...')  
    except:
        print('Explainer Create Error !\n{}'.format(traceback.format_exc()))
        return None
    
    Explainer = MLExplainer(model_id, explainer_object, explainer_config, model_parameters)
    
    return Explainer    
    
       
def get_explainer_values_task(DataFrame, Explainer=None, Model=None, explainer_config={'Method':'shap'}, verbose=False):
    """    
    Parameters
    ----------    
    DataFrame : pandas.DataFrame
    Explainer : mltk.MLExplainer
    Model : mltk.MLModel
    explainer_config : dict or json         
    
    Returns
    -------  
    ShapValues : pandas.DataFrame
    VariableValues : pandas.DataFrame
    
    """
    
    if Explainer==None and Model!=None:
        Explainer = build_explainer(Model, explainer_config)
    elif Explainer!=None and Model==None:
        Explainer = Explainer
    else:
        print('explainer or materials to create explainer not provided !')
        
    model_variables = Explainer.get_model_parameter('ModelVariables')
    model_type = Explainer.get_model_parameter('ModelType')
    predicted_value_column = Explainer.get_model_parameter('PredictedValueColumn')    
    id_columns = Explainer.get_explainer_config_item('IdColumns')
    class_number = Explainer.get_explainer_config_item('ClassNumber')
    method = Explainer.get_explainer_config_item('Method')
    explainer = Explainer.get_explainer_object()
    fill_missing = Explainer.get_explainer_config_item('FillMissing')
    
    # Blanck columns for non-existance variables
    missing_variables = list(set(model_variables) - set(DataFrame.columns)) # Find columns not found in the dataset
    
    for f in missing_variables:
        DataFrame[f]=fill_missing
        if verbose:
            print('Column [{}] does not exist in the dataset. Created new column and set to {}...\n'.format(f,missing_variables))
            
            
    if method == 'shap':
        ImpactValues, VariableValues = get_shap_values(DataFrame, explainer, 
                                            model_variables=model_variables, 
                                            id_columns=id_columns, 
                                            model_type=model_type, 
                                            class_number=class_number, 
                                            predicted_value_column=predicted_value_column
                                            )
    else:
        ImpactValues = None
        VariableValues = None
    
    return ImpactValues, VariableValues

def get_explainer_visual(ImpactValues, VariableValues, Explainer, visual_config={'figsize':[20,4]}):
    """    
    Parameters
    ----------  
    ImpactValues : pandas.DataFrame
    VariableValues : pandas.DataFrame
    Explainer : mltk.MLExplainer
    visual_config : dict
        
    Returns
    -------  
    explainer_visual : matplotlib.figure.Figure
    """
    
    base_value = Explainer.get_base_value()
    model_variables = Explainer.get_model_variables()
    method = Explainer.get_explainer_method()
    
    if method == 'shap':
        figsize = visual_config['figsize']
        text_rotation = visual_config['text_rotation']
        explainer_visual = get_shap_force_plot(ImpactValues, VariableValues, model_variables, base_value, figsize=figsize, text_rotation=text_rotation)
    else:
        explainer_visual = None
        print('method {} not implemented'.format(method))
    
    return explainer_visual

def get_shap_values(DataFrame, explainer, model_variables, id_columns=[], model_type='classification', class_number=0, predicted_value_column=None):
    """    
    Parameters
    ----------    
    DataFrame : pandas.DataFrame
    explainer : shap.Explainer
    model_variables : list(str)
    id_columns : list(str)
    model_type : {'regression', 'classification'}
    class_number : int
    predicted_value_column : str
    
    Returns
    -------  
    ShapValues : pandas.DataFrame
    VariableValues : pandas.DataFrame
    """
    
    DataFrame = DataFrame.reset_index(drop=True)
    
    try:
        if model_type=='classification':
            shap_values = explainer.shap_values(DataFrame[model_variables])[class_number] 
            base_value = explainer.expected_value[class_number]
        elif model_type=='regression':
            shap_values = explainer.shap_values(DataFrame[model_variables])
            base_value = explainer.expected_value[0]
        # To DataFrame
        ShapValues = pd.DataFrame(data=shap_values, columns=model_variables)
        ShapValues['BaseValue'] = base_value
        del(shap_values)
    except:
        print('Error creating Shap Values\n{}'.format(traceback.format_exc()))    
        return None
    
    try:
        for c in id_columns:
            ShapValues[c] = DataFrame[c].values
    except:
        print('Error creating Id columns\n{}'.format(traceback.format_exc()))     
    
    try:
        ShapValues['OutputValue'] = DataFrame[predicted_value_column].values
    except:
        ShapValues['OutputValue'] = None
        print('Error creating OutputValue columns\n{}'.format(traceback.format_exc()))     
        
    return ShapValues[id_columns+model_variables+['BaseValue', 'OutputValue']],  DataFrame[id_columns+model_variables]#Pandas Dataframe


def get_shap_force_plot(ShapValues, VariableValues, model_variables, base_value, figsize=(20, 3), text_rotation=90):
    """
    Parameters
    ----------  
    ShapValues : pandas.DataFrame
    VariableValues : pandas.DataFrame
    model_variables : list(str)
    figsize : (int, int), default (20, 3)
    text_rotation : float, default 90
    
    Returns
    -------  
    shap_force_plot : matplotlib.figure.Figure
    """

    shap_force_plot = shap.force_plot(
        base_value=base_value,
        shap_values=ShapValues[model_variables].values,
        features=VariableValues[model_variables].values,
        feature_names=model_variables,
        out_names=None,
        link='identity',
        plot_cmap='RdBu',
        matplotlib=True,
        show=False,
        figsize=figsize,
        ordering_keys=None,
        ordering_keys_time_format=None,
        text_rotation=text_rotation,
    )
    
    return shap_force_plot

def get_shap_impact_plot(ShapValues, VariableValues, model_variables, iloc=0, top_n=5):
    """
    Parameters
    ----------  
    ShapValues : pandas.DataFrame
    VariableValues : pandas.DataFrame
    model_variables : list(str)
    iloc : int, default 0
    top_n : int, default 5
    
    Returns
    -------  
    figure : matplotlib.figure.Figure
    """    
    
    impact_values = ShapValues[model_variables].iloc[iloc]
    variable_values = VariableValues[model_variables].iloc[iloc]
    variable_values.name = 'Value'
    
    posivite_impact_values = impact_values[impact_values>=0.0].sort_values(ascending=False)
    posivite_impact_values.name = 'Impact'

    negative_impact_values = impact_values[impact_values<=0.0].sort_values(ascending=True)
    negative_impact_values.name = 'Impact'

    figure, axes = plt.subplots(nrows=1, ncols=2)
    negative_impact_values.head(top_n)[::-1].plot(kind='barh', ax=axes[0], color='r', legend=False)
    posivite_impact_values.head(top_n)[::-1].plot(kind='barh', ax=axes[1], color='g', legend=False)
    axes[1].yaxis.tick_right()
    
    return figure

def get_shap_impact_summary(ShapValues, VariableValues, model_variables, base_value=None, iloc=0, top_n=None, show_plot=False):
    """
    Parameters
    ----------  
    ShapValues : pandas.DataFrame
    VariableValues : pandas.DataFrame
    model_variables : list(str)
    iloc : int, default 0
    top_n : int, default None
    
    Returns
    -------  
    ImpactSummary : pandas.DataFrame
    """        
    impact_values = ShapValues[model_variables].iloc[iloc]
    variable_values = VariableValues[model_variables].iloc[iloc]
    variable_values.name = 'Value'
    
    posivite_impact_values = impact_values[impact_values>=0.0].sort_values(ascending=False)
    posivite_impact_values.name = 'Impact'

    negative_impact_values = impact_values[impact_values<=0.0].sort_values(ascending=True)
    negative_impact_values.name = 'Impact'

    sum_posivite_impact_values = posivite_impact_values.sum()
    sum_negative_impact_values = negative_impact_values.sum()

    norm_posivite_impact_values = posivite_impact_values/sum_posivite_impact_values
    norm_negative_impact_values = negative_impact_values/sum_negative_impact_values
    norm_posivite_impact_values.name = 'NormalizedImpact'
    norm_negative_impact_values.name = 'NormalizedImpact'

    if base_value != None:
        output_value = sum_posivite_impact_values + sum_negative_impact_values + base_value
    else:
        output_value = None
        
    out_posivite_impact_values = pd.concat([posivite_impact_values, norm_posivite_impact_values, variable_values], join='inner', axis=1)
    out_posivite_impact_values['Sign'] = 'P'
    out_posivite_impact_values['Rank'] = np.arange(out_posivite_impact_values.shape[0])+1
    out_posivite_impact_values['Output'] = output_value

    out_negative_impact_values = pd.concat([negative_impact_values, norm_negative_impact_values, variable_values], join='inner', axis=1) 
    out_negative_impact_values['Sign'] = 'N'
    out_negative_impact_values['Rank'] = np.arange(out_negative_impact_values.shape[0])+1
    out_negative_impact_values['Output'] = output_value

    print('sum_posivite_impact_values', sum_posivite_impact_values)
    print('sum_negative_impact_values', sum_negative_impact_values)

    if show_plot:
        get_shap_impact_plot(ShapValues, VariableValues, model_variables, iloc=iloc, top_n=top_n)
    
    if top_n==None:    
        return out_posivite_impact_values.append(out_negative_impact_values)
    else:
        return out_posivite_impact_values.head(top_n).append(out_negative_impact_values.head(top_n))
    
def get_explainer_report(DataFrame, Explainer, top_n=5, show_plot=False, return_type='frame'):
    """    
    Parameters
    ----------    
    DataFrame : pandas.DataFrame
    Explainer : mltk.MLExplainer
    top_n : int, default 5
    show_plot : bool, defualt False
    return_type : {'frame', 'json'}, defua;t 'frame'
    
    Returns
    -------  
    ImpactSummary : pandas.DataFrame
    impact_plot : matplotlib.figure.Figure (optional)
    """

    ImpactValues, VariableValues = get_explainer_values_task(DataFrame, Explainer=Explainer)
    
    model_variables = Explainer.get_model_variables()
    base_value = Explainer.get_base_value()
    
    ImpactSummary = get_shap_impact_summary(ImpactValues, VariableValues, model_variables, base_value=base_value, iloc=0, top_n=top_n)

    if show_plot:
        impact_plot = get_shap_impact_plot(ImpactValues, VariableValues, model_variables, iloc=0, top_n=top_n)

    if return_type=='frame':
        if show_plot:
            return ImpactSummary, impact_plot
        else:
            return ImpactSummary
    elif return_type=='json':
        if show_plot:
            return ImpactSummary.to_json(orient='columns'), impact_plot
        else:
            return ImpactSummary.to_json(orient='columns')
    else:
        print("return type is set to 'frame'. use 'json' to return json output")
        if show_plot:
            ImpactSummary, impact_plot
        else:
            return ImpactSummary