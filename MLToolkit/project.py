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
import datetime
from mltk import *

class MLSetup():
    def __init__(self, id, name, model_id=None, data_item_id=None):
        self.id = id
        self.name = name
        self.model_id = model_id
        self.data_item_id = data_item_id
        
    def get_setup_id(self):
        return self.id 

    def get_setup_name(self):
        return self.name 

    def get_model_id(self):
        return self.model_id 

    def get_data_item_id(self):
        return self.data_item_id 
    
    def set_model_id(self, model_id):
        self.model_id = model_id

    def set_data_item_id(self, data_item_id):
        self.data_item_id = data_item_id 

    
class MLProject():
    def __init__(self, id=1, name='test', attributes={}, config={}):
        self.id = id
        self.name = name
        self.attributes = attributes
        self.config = config
        ####################
        self.data_store = {}
        self.model_shelf = {}
        self.work_bench = {}        
        self.active_model_id = None
        self.active_data_item_id = None
        self.active_mlsetup_id = None
                
    ##[ Project ]##############################################################
    def load_project(self):
        return None
    
    def save_project(self):
        return None
    
    ##[ Work Bench ]###########################################################
    
    def add_mlsetup(self, set_up):   
        self.work_bench[set_up['setup_id']] = set_up

    def create_mlsetup(self, setup_id=1, setup_name='test', model_id=None, data_item_id=None, feature_variables=None):
        set_up = {'setup_id':setup_id, 'setup_name':setup_name, 'model_id':model_id, 'data_item_id':data_item_id, 'feature_variables':feature_variables}
        self.work_bench[set_up['setup_id']] = set_up
        return set_up        

    def update_mlsetup_feature_variables(self, setup_id, feature_variables):   
        self.work_bench[setup_id]['feature_variables'] = feature_variables

    def update_mlsetup_model_id(self, setup_id, model_id):   
        self.work_bench[setup_id]['model_id'] = model_id

    def update_mlsetup_data_item_id(self, setup_id, data_item_id):   
        self.work_bench[setup_id]['data_item_id'] = data_item_id
        
    def remove_mlsetup(self, setup_id):   
        del self.work_bench[setup_id]
        
    ##[ Model_Shelf ]##########################################################
    def get_model_list(self):
        model_list = {}
        for model_id in self.model_shelf.keys():
            Model = self.model_shelf[model_id]
            ml_algorithm = Model.get_model_algorithm()
            roc_auc = Model.get_auc(curve='roc')
            prc_auc = Model.get_auc(curve='prc')
            model_list[model_id] = {'model_id':model_id, 'ml_algorithm':ml_algorithm, 'roc_auc':roc_auc, 'prc_auc':prc_auc }
        return model_list
    
    def add_model(self, Model, set_active=True):   
        self.model_shelf[Model.get_model_id()] = Model
        
    def remove_model(self, model_id): 
        del self.model_shelf[model_id]
        
    def get_model(self, model_id):
        return self.model_shelf[model_id]
        
    def get_active_model(self):
        return self.model_shelf[self.active_model_id]
          
    # Data_Store
    def get_data_item_list(self):
        data_item_list = {}
        for data_item_id in self.data_store.keys():
            Data = self.data_store[data_item_id]
            column_dict = Data.dtypes.astype('str').to_dict()
            column_count = len(Data.columns)
            row_count = len(Data.index)
            data_item_list[data_item_id] = {'data_item_id':data_item_id, 'column_count':column_count, 'row_count':row_count, 'column_dict':column_dict}
        return data_item_list        
        
    def add_data_item(self, DataFrame, data_item_id='test'):   
        self.data_store[data_item_id] = DataFrame
        
    def update_data_item(self, DataFrame, data_item_id=None):   
        self.data_store[data_item_id] = DataFrame
        
    def remove_data_item(self, data_item_id): 
        del self.data_store[data_item_id]

    def get_data_item(self, data_item_id):
        return self.data_store[data_item_id]

    def get_active_data_item(self):
        return self.data_store[self.active_data_item_id]

    def get_active_data_item_id(self):
        return self.active_data_item_id
    
    def set_active_data_item(self, data_item_id):
        self.active_data_item_id = data_item_id
    
    # Process    
    def load_data(self, load_data_dict, set_active=False):
        """
        Parameters
        ----------
        load_data_json: json str
        
        Returns
        -------
        None
        
        """
        DataFrame, data_name = load_data_task(load_data_dict, return_name=True)
        data_item_id = data_name+'_'+datetime.now().strftime('%Y%m%d%H%M%S') 
            
        self.add_data_item(DataFrame, data_item_id=data_item_id) 
        
        if set_active:
            self.active_data_item_id = data_item_id
        
    def save_data(self, save_data_json):
        type = save_data_json['type']
        if type=='csv':
            print('Not implemented')
    
    def create_features(self, variables_setup_dict, data_item_id=None, use_active_data_item=False):
        
        if data_item_id==None and use_active_data_item:
            data_item_id = self.get_active_data_item_id()
            
        DataFrame = self.get_data_item(data_item_id)
        
        # Setup feature variables
        DataFrame, category_variables, binary_variables, target_variable  = setup_variables_task(DataFrame, variables_setup_dict)

        # Create One Hot Encoded Variables
        DataFrame, feature_variables, target_variable = to_one_hot_encode(DataFrame, category_variables=category_variables, binary_variables=binary_variables, target_variable=target_variable)
        
        self.update_data_item(DataFrame, data_item_id=data_item_id)
        
    def fit_model(self, model_setup_dict, data_item_id=None, use_active_data_item=False, set_active=True, variables_setup_dict=None):
        
        if data_item_id==None and use_active_data_item:
            data_item_id = self.get_active_data_item_id()
            
        DataFrame = self.get_data_item(data_item_id)        
        Model = build_ml_model_task(DataFrame, model_setup_dict, variables_setup_dict)        
        self.add_model(Model, set_active=set_active)
        
        return Model 
       
    def evaluate_model(self):
        return None
        
    def save_model(self):
        return None
        
    def load_model(self):    
        return None

