__name__="mltk"

"""
MLToolkit - a verstile helping library for machine learning
===========================================================
'MLToolkit' is a Python package providing a set of user-friendly functions to 
help building machine learning models in data science research or production 
focused projects. It is compatible with and interoperate with popular data 
analysis, manipulation and machine learning libraries Pandas, Sci-kit Learn, 
Tensorflow, Statmodels, Catboost, XGboost, etc.

Author
------
- Sumudu Tennakoon

Links
-----
Website: http://sumudu.tennakoon.net/projects/MLToolKit
Github: https://mltoolkit.github.io/MLToolKit

License
-------
Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
import json
import traceback
import numpy as np
import configparser
import os
import requests
from mltk.model import *


class ModelChest():
    def __init__(self, config_file=None, model_chest_file_path=None, model_objects_folder=None, model_folder=None, api_address=None):
        """
        Model Chest Class
               
        Parameters
        ----------
        config_file: str, optional, default None
        model_chest_file_path: str, optional, default None
        model_objects_folder: str, optional, default None
        model_folder : stroptional, default None
        api_address : str, optional, default None
        
        Returns
        -------
        model_chest_json: dict
        """ 
        self.config_path = config_file
        self.model_chest_file_path = model_chest_file_path
        self.model_objects_folder=model_objects_folder
        self.api_address=api_address
        self.model_folder=model_folder
        
        if config_file != None:
            self.config_path=config_file
            config = configparser.ConfigParser()
            config.read(config_file)
            api_home_folder = config['FOLDERS']['APIHomeFolder'] 
            model_folder = config['FOLDERS']['ModelFolder']
            if model_folder[0]=='*':                
                self.model_folder = os.path.join(api_home_folder, model_folder[1:])
            else:
                self.model_folder = os.path.join(api_home_folder, model_folder)                
            self.model_chest_file_path = os.path.join(self.model_folder, config['MODEL']['ModelFile'])
            self.model_chest_json = self.read_model_chest()
            folder = self.model_chest_json['Folder']
            if folder[0]=='*':                
                self.model_objects_folder = os.path.join(self.model_folder, folder[1:])    
            else:
                self.model_objects_folder = folder
            
            try:
                api_address = config['API']['HostAddress']
                port = config['API']['Port']
                if len(port)>0:
                    self.api_address = '{}:{}'.format(api_address,port)
                else:
                    self.api_address = api_address
            except:
                print('Error in reading host adress !')
        elif model_chest_file_path != None and  model_objects_folder != None:
            self.model_chest_json = self.read_model_chest()
        else:
            print('Required either [config_path] or [model_chest_file_path, model_folder]')
            self.model_chest_json = self.create_model_chest()
            
    def create_model_chest(self, model_folder=None, model_chest_file_path=None):
        """       
        Parameters
        ----------
        model_folder: str
        model_chest_file_path: str
        
        Returns
        -------
        model_chest_json: dict
        """ 
        
        model_chest_json = {
                'Folder': '*ModelsChest', 
                'Models': {}, 
                'Default': None
                }
        
        if model_folder !=None:
            self.model_folder = model_folder   
        elif self.model_folder == None:
            self.model_folder = os.getcwd()
        else:
            self.model_folder = self.model_folder
            
        if model_chest_file_path != None:
            self.model_chest_file_path = model_chest_file_path   
        elif self.model_chest_file_path == None:
            self.model_chest_file_path = os.path.join(os.getcwd() , 'model_chest.json')             
        else:
            self.model_chest_file_path = self.model_chest_file_path
            
        self.model_objects_folder = os.path.join(self.model_folder, 'ModelsChest')
        print('Model Chest Folder set to :\n{}'.format(self.model_objects_folder))        
        print('Created new Model Chest :\n{}'.format(json.dumps(model_chest_json, indent=4)))
        
        return model_chest_json        
        
    def read_model_chest(self):
        """
        Loads Model Chest 
        
        Parameters
        ----------
        save_path: str
        
        Returns
        -------
        model_chest_json: dict
        """ 
        model_chest_file_path = self.model_chest_file_path
        try:
            with open(model_chest_file_path, 'r') as model_chest_file:
                model_chest_json = json.load(model_chest_file)
        except FileNotFoundError:
            print("Model Chest JSON File {} not found.".format(self.model_chest_file_path)) 
        
        self.model_chest_json = model_chest_json
        return model_chest_json
    
    def save_model_chest(self, save_path=None):
        """
        Save Model Chest 
    
        Parameters
        ----------
        save_path: str
        
        Returns
        -------
        None
        """  
        if save_path==None:
            model_chest_file_path = self.model_chest_file_path
        else:
            model_chest_file_path = save_path
            self.model_folder = os.path.dirname(save_path)
            
        model_chest_json = self.model_chest_json

        if model_chest_file_path==None:
            print('Model Chest Save Path is None. Retry with save_path=<<path_to_file>>')
        else:        
            try:
                # Create new folder if not exists
                folder = self.model_chest_json['Folder']
                if folder[0]=='*':                
                    self.model_objects_folder = os.path.join(self.model_folder, folder[1:])    
                else:
                    self.model_objects_folder = folder
                    
                if not os.path.exists(self.model_objects_folder):
                    print('Model Chest folder not found. New folder {} created.'.format(self.model_objects_folder))
                    os.makedirs(self.model_objects_folder)
    
                with open(model_chest_file_path, 'w') as model_chest_file:
                    json.dump(model_chest_json, model_chest_file, indent=4)
                    
                print('Model Chest saved to {}'.format(model_chest_file_path))
            except FileNotFoundError:
                print("Error writing Model Chest JSON File {}.".format(self.model_chest_file_path))         
        
    def get_model_chest_json(self):
        """
        Parameters
        ----------
        None
        
        Returns
        -------
        model_chest_json : dict
        """  
        return self.model_chest_json
    
    def add_model(self, model_key, model_file=None, model_object=None, replace=False):
        """
        Parameters
        ----------
        model_key : str
        model_file : str, default None
        model_object : str, default None
        replace : str, default False
        
        Returns
        -------
        None
        """  
        try:
            if (model_key in self.model_chest_json['Models'].keys()) and replace==False:
                print('Model key "{}" already excist in the chest. Try again with "replace=True" to replace'.format(model_key))                
            else:
                try:
                    if model_object != None:
                        model_object_folder = self.model_objects_folder
                        model_file = '{}.pkl'.format(model_object.get_model_id())

                        save_file_path = os.path.join(model_object_folder, model_file)
                        
                        if not os.path.exists(self.model_objects_folder):
                            print('Model Chest folder not found. New folder {} created.'.format(self.model_objects_folder))
                            os.makedirs(self.model_objects_folder)
                    
                        save_model(model_object, save_file_path)
                except:
                    print('ERROR: \n{}'.format(traceback.format_exc()))
                    
                self.model_chest_json['Models'][model_key]=model_file
                print('Model "{}":"{}" added'.format(model_key, model_file))

                if len(self.model_chest_json['Models'].keys()) == 1:
                    self.set_default_model(model_key)
        except:
            print('Error adding Model "{}":"{}"'.format(model_key, model_file))
            
    def remove_model(self, model_key):
        """
        Parameters
        ----------
        model_key : str
        
        Returns
        -------
        None
        """  
        try:
            del self.model_chest_json['Models'][model_key] 
            print('Model "{}" removed'.format(model_key))
        except:
            print('Error removing Model "{}"'.format(model_key))
 
    def get_model_keys(self):
        """
        Parameters
        ----------
        None
        
        Returns
        -------
        model_keys : list(str)
        """  
        return list(self.model_chest_json['Models'].keys())
    
    def set_default_model(self, model_key):
        """
        Parameters
        ----------
        model_key : str
        
        Returns
        -------
        None
        """  
        self.model_chest_json['Default'] = model_key
        print('Model "{}" set as Default'.format(model_key))
        
    def get_model_object(self, model_key):
        """
        Parameters
        ----------
        model_key : str
        
        Returns
        -------
        MLModelObject : mltk.MlModel
        """  
        model_object_file = os.path.join(self.model_objects_folder, self.model_chest_json['Models'][model_key])
        return load_model(model_object_file)
    
    def reload_api_models(self, api_address=None, reload_endpoint='reload_models', key=None):
        """
        Parameters
        ----------
        api_address : str
        reload_endpoint : str, default 'reload_models'
        key : str, default None
        
        Returns
        -------
        response : str
        """  
        reload_endpoint = reload_endpoint
        
        if api_address==None:
            api_address = self.api_address
        
        url = 'http://{}/{}?key={}'.format(api_address, reload_endpoint, key)
        
        response = requests.get(url)
        
        return response.text
        