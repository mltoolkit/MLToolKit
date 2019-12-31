# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 2019
@Last update: Sun July 01 2019
@author: Sumudu Tennakoon
@licence:
   Copyright 2019 Sumudu Tennakoon

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
@notice: 
    If you use part of this code, concepts or code pattern, please 
    kindly give credits when necessary.    
   
"""
import requests
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import traceback

class MLToolkitAPISDK:
    
    def score(self, input_source=None, input_type='json', method='POST', url="http://127.0.0.1:5000/ml_api_json", user_id='TestUser', reference=' TestReference', dict_ouput=False):
        
        if input_type == 'json':
            url="http://127.0.0.1:5000/ml_api_json"
            try:
                form_data = {
                    'UserId': user_id,
                    'Reference': reference,
                    'InputData': input_source
                    }    
                if method=='POST':
                    response = requests.post(url, data=form_data)
                elif method=='GET':
                    response = requests.get(url, params=form_data)
            except:
                response = print(format(traceback.format_exc()))                
                
        elif input_type == 'file':
            url="http://127.0.0.1:5000/ml_api_file"
            try:
                data_filename = os.path.basename(input_source)
                files_data = {
                        'file': (data_filename, open(input_source, 'rb'))       
                    }   
                form_data = {
                    'UserId': user_id,
                    'Reference': reference,
                    } 
                if method=='POST':
                    response = requests.post(url, files=files_data, data=form_data)
                elif method=='GET':
                    response = requests.get(url, files=files_data, params=form_data)
            except:
                response = print(format(traceback.format_exc()))
        else:
            response = None      
        
        try:
            if dict_ouput==True:
                response = json.loads(response.text)
        except:
            response = print(format(traceback.format_exc()))
            
        return response    