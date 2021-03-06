# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 2019
@Last update: Sun August 01 2019
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

import flask
from flask.json import JSONEncoder
from flask import Flask, Response, request, redirect, url_for, flash, send_file, send_from_directory, render_template, Markup
from werkzeug.utils import secure_filename
import os
import io
import json 
import sys
import configparser
import argparse
import traceback
import csv
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mltk as mltk

###############################################################################
def generate_config_file(): 
    cwd = os.getcwd()
    try:
        os.mkdir('API')
        os.mkdir('API/static')
        os.mkdir('API/templates')
    except:
        print('ERROR: \n{}'.format(traceback.format_exc()))
    	
    try:  
        if not os.path.exists('API/config.ini'):	
            config = configparser.ConfigParser()    
            config['DEFAULT']['Application'] = '{Application}'
            config['DEFAULT']['OutputFilePrefix'] = '{OutputFilePrefix}'
            config.add_section('FOLDERS') ###
            config['FOLDERS']['APIHomeFolder'] = '{api_home_folder}'
            config['FOLDERS']['ModelFolder'] = '{model_folder}'
            config['FOLDERS']['DropBoxFolder'] = '{dropbox_folder}'
            config['FOLDERS']['TempFolder']  = '{temp_folder}'          
            config['FOLDERS']['SampleFileFolder'] = '{sample_folder}'
            config.add_section('SAMPLES') ###
            config['SAMPLES']['JSONTestFile'] = 'test.json'
            config['SAMPLES']['BatchInputTestFile'] = 'test.csv'
            config.add_section('MODEL') ###
            config['MODEL']['ModelFile']='{model_file}'
            config['MODEL']['ETLPyScript'] = 'ETL.py'
            config['MODEL']['PostProcessPyScript'] = 'PostProcess.py'
            config.add_section('API') ###
            config['API']['HostAddress'] = '127.0.0.1'
            config['API']['Port'] = '5000'
            config['API']['URLRecordLimit'] = '1'
            config['API']['Static'] = 'static'
            config['API']['HTMLTemplates'] = 'templates'
            config.add_section('KEYS') ###
            config['KEYS']['ConfigAccessKey'] = 'MLTKCONFIG1234'
            config['KEYS']['APIAccessKey'] = 'MLTKAPI1234'            
    
            with open('API/config.ini', 'w') as configfile: 
                config.write(configfile)
    except:
        print('ERROR: \n{}'.format(traceback.format_exc()))
        sys.exit(1)
            
    return os.path.join('{}'.format(cwd),'API')

def model_test_run(model_input_data_json=None, model_input_data_file=None):
    print('-'*80)    
    try:
        with open(model_input_data_json) as json_file:  
            input_data = json.load(json_file)
        input_data_str = json.dumps(input_data)
        #edges = MLModelObject.score_parameters['Edges']
        #output = mltk.score_records(input_data_str, MLModelObject, edges, ETL=ETL, return_type='json')
        model_key = default_model_key
        output = score_routine(input_data_str, model_key=model_key, ETL=ETL, return_type='json')
        print('* Scoring Test JSON format: DONE')
        print('   RESULT:\n{}'.format(output))  
    except:
        print('* Scoring Test JSON format: FAILED')
        print('ERROR: {}'.format(traceback.format_exc()))
    print('-'*80)

    try:
        model_input_data = pd.read_csv(model_input_data_file).head(3)
        #output = mltk.score_records(model_input_data, MLModelObject, edges, ETL=ETL, return_type='frame')
        model_key = default_model_key
        output = score_routine(model_input_data, model_key=model_key, ETL=ETL, return_type='frame')
        print(' * Scoring Test CSV FILE format: DONE')
        print('   RESULT:\n{}'.format(output.transpose()))  
    except:
        print(' * Scoring Test CSV FILE format: FAILED')
        print('ERROR: {}'.format(traceback.format_exc()))
    print('-'*80)
    
###############################################################################
# CONFIG FILE PATH
try:
    parser = argparse.ArgumentParser(description='MLToolkit Model Server')
    parser.add_argument('config', help='path to config.ini')
    args = parser.parse_args()
    config_file=args.config
except:
    print('ERROR 101: Input error. No : \n{}'.format(traceback.format_exc()))     
    try:
        config_file = input("Path to config.ini file (Press enter to generate file template): ")
        if len(config_file)==0:
            raise Exception('No config.ini file path given... genrating template...')
    except:
        print('ERROR 101: Input error. No : \n{}'.format(traceback.format_exc())) 
        api_folder = generate_config_file()
        print('API folder and config.ini file template generated at {}.\nEdit the config.ini template before strating the model server'.format(api_folder))
        sys.exit(1) 
        
print('config_file:', config_file)        
###############################################################################
# LOAD CONFIG
try:
    config = configparser.ConfigParser()
    config.read(config_file)
    application_name = config['DEFAULT']['Application']
    out_file_prefix = config['DEFAULT']['OutputFilePrefix']
    # FOLDERS
    api_home_folder = config['FOLDERS']['APIHomeFolder']
    model_folder = config['FOLDERS']['ModelFolder'] 
    if model_folder[0]=='*':
        model_folder = os.path.join(api_home_folder, model_folder[1:])
    dropbox_folder = config['FOLDERS']['DropBoxFolder'] 
    if dropbox_folder[0]=='*':
        dropbox_folder = os.path.join(api_home_folder, dropbox_folder[1:])
    sample_folder = config['FOLDERS']['SampleFileFolder'] 
    if sample_folder[0]=='*':
        sample_folder = os.path.join(api_home_folder, sample_folder[1:])
    temp_folder = config['FOLDERS']['TempFolder'] 
    if temp_folder[0]=='*':
        temp_folder = os.path.join(api_home_folder, temp_folder[1:])
    # SAMPLES
    model_input_data_json = os.path.join(sample_folder, config['SAMPLES']['JSONTestFile'])
    model_input_data_file = os.path.join(sample_folder, config['SAMPLES']['BatchInputTestFile'])
    # MODEL
    model_file = os.path.join(model_folder, config['MODEL']['ModelFile']) 
    etl_py_script = os.path.join(model_folder, config['MODEL']['ETLPyScript'])
    post_process_py_script = os.path.join(model_folder, config['MODEL']['PostProcessPyScript'])
    # API   
    host_address = config['API']['HostAddress'] #'127.0.0.1'
    port = config['API']['Port'] #'5000'    
    record_limit = int(config['API']['URLRecordLimit'])
    static_folder = os.path.join(api_home_folder, config['API']['Static'])
    html_template_folder = os.path.join(api_home_folder, config['API']['HTMLTemplates'])  
    # KEYS
    config_access_key = config['KEYS']['ConfigAccessKey']
    api_access_key = config['KEYS']['APIAccessKey']            
except:
    print('ERROR 102: Confuguration file Config.ini not found: \n{}'.format(traceback.format_exc()))
    sys.exit(1) 

print('-'*80) 
print('ML TooKit Model Server Confuguarion Profile...')
print('* api_home_folder:', api_home_folder) 
print('* model_folder:', model_folder) 
print('* dropbox_folder:', dropbox_folder) 
print('* temp_folder:', temp_folder) 
print('* sample_folder:', sample_folder)    
print('* static_folder:', static_folder)
print('* html_template_folder:', html_template_folder)                
print('* model_file:', model_file)        
print('* etl_py_script:', etl_py_script)   
print('* post_process_py_script:', post_process_py_script) 
print('* record_limit:', record_limit)           
print('* test_input_data_json:', model_input_data_json)      
print('* test_input_data_file:', model_input_data_file)        
print('-'*80)
    
# INITIALIZE FLASK APP
app = flask.Flask(__name__, template_folder=html_template_folder, static_folder=static_folder)  

###############################################################################    
ModelFile = os.path.join(model_folder,model_file)
AcceptedFileTypes = set(['csv']) #, 'json', 'xlsx', 'xls'])

###############################################################################    
class custom_json_encoder(JSONEncoder):
    def default(self, obj):
        try:
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return round(float(obj),4)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            iterable = iter(obj)
        except TypeError:
            print('ERROR: \n{}'.format(traceback.format_exc()))
        else:
            return list(iterable)
        return JSONEncoder.default(self, obj)  
    
###############################################################################       
def is_accepted_file_type(file_name):
    return os.path.splitext(file_name)[1][1:] in AcceptedFileTypes    

###############################################################################
# Jsonifying the response object with Cross origin support [1].
def send_response(responseObj):
    response = flask.jsonify(responseObj)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
    response.headers.add('Access-Control-Allow-Headers', 'accept,content-type,Origin,X-Requested-With,Content-Type,access_token,Accept,Authorization,source')
    response.headers.add('Access-Control-Allow-Credentials', True)
    return response

###############################################################################
def score_routine1(model_input_data, model_key=None, ETL=None, return_type='frame', return_model_key=False):
      
    if model_key==None:
        model_input_data = json.loads(model_input_data)
        try:
            model_key = model_input_data['Model']
        except:
            model_key = model_input_data['model']
    #Generate scores
    edges = MLModelObjects[model_key].score_parameters['Edges']
    # model_input_data will be passed to Application ETL function to prepare the dataset to model.
    scored_data = mltk.score_records(model_input_data, MLModelObjects[model_key], edges=edges, ETL=ETL, return_type='frame')
    # Execute post-processing (save to file, upload to DB server, etc)
    file_reference = str(uuid.uuid4())
    score_result_columns = [MLModelObjects[model_key].get_score_variable(), MLModelObjects[model_key].get_score_label()] # Mandatory field to save
    output = PostProcess(scored_data, score_result_columns, file_reference, dropbox_folder, temp_folder) # Should return a dataframe
    
    if return_type=='dict':
        output = output.to_dict(orient='records') # return dict
    elif return_type=='json':
        output = output.to_json(orient='records') # return JSON 
    else:
        pass # Keep as data frame
    
    if return_model_key:
        return output, model_key
    else:
        return output  
    
###############################################################################    
def score_routine(model_input_data, model_key=None, ETL=None, return_type='frame', return_model_key=False):
    
    if model_key==None:
        model_input_data = json.loads(model_input_data)
        try:
            model_key = model_input_data['Model']
        except:
            model_key = model_input_data['model']    
    else:
        pass
    
    try:
        model_key = model_key.replace(" ", "").split(',')  
    except:
        pass
    
    if len(model_key)==1:
        model_key = model_key[0]
        
    model_results_index = {} 
    
    if type(model_key)==list:
        
        keep_columns = []
        input_columns = []
        
        for mkey in model_key:
            #Generate scores
            edges = MLModelObjects[mkey].get_score_edges()
            score_variable = '{}{}'.format(MLModelObjects[mkey].get_score_variable(), mkey)
            score_label = '{}{}'.format( MLModelObjects[mkey].get_score_label(), mkey)
            # Score
            if len(keep_columns)==0:
                scored_data, input_columns = mltk.score_records(model_input_data, MLModelObjects[mkey], edges=edges, 
                                                                score_label=score_label, score_variable=score_variable, 
                                                                ETL=ETL, return_type='frame', return_input=True)
            else:
                scored_data = mltk.score_records(scored_data[input_columns+keep_columns], MLModelObjects[mkey], edges=edges, 
                                                                score_label=score_label, score_variable=score_variable, 
                                                                ETL=ETL, return_type='frame', return_input=False)
            model_results_index[mkey]=[score_variable, score_label]
            keep_columns = keep_columns + [score_variable, score_label]
            
        score_result_columns = keep_columns
        scored_data = scored_data[input_columns+score_result_columns]  
          
    else:
        #Generate scores
        edges = MLModelObjects[model_key].get_score_edges()
        score_variable = MLModelObjects[model_key].get_score_variable()
        score_label = MLModelObjects[model_key].get_score_label()
            
        # model_input_data will be passed to Application ETL function to prepare the dataset to model.
        scored_data = mltk.score_records(model_input_data, MLModelObjects[model_key], edges=edges, ETL=ETL, return_type='frame')
        model_results_index[model_key]=[score_variable, score_label]
        
        score_result_columns = [MLModelObjects[model_key].get_score_variable(), MLModelObjects[model_key].get_score_label()] # Mandatory field to save
        
    # Execute post-processing (save to file, upload to DB server, etc)
    file_reference = str(uuid.uuid4())
    post_process_parameters = {'model_results_index':model_results_index}
    output = PostProcess(scored_data, score_result_columns, file_reference, dropbox_folder, temp_folder, post_process_parameters) # Should return a dataframe
        
    if return_type=='dict':
        output = output.to_dict(orient='records') # return dict
    elif return_type=='json':
        output = output.to_json(orient='records') # return JSON 
    else:
        pass # Keep as data frame
    
    if return_model_key:
        return output, model_key
    else:
        return output  
###############################################################################
    
@app.after_request
def add_header(response):
    '''
    Fixing browser chache issues. You can read more about these from
    # https://stackoverflow.com/questions/12034949/flask-how-to-get-url-for-dynamically-generated-image-file
    # https://stackoverflow.com/questions/23112316/using-flask-how-do-i-modify-the-cache-control-header-for-all-output
    # https://stackoverflow.com/questions/13768007/browser-caching-issues-in-flask
    '''
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

@app.route('/')
def index():
    URL = 'http://{}:{}/input_form'.format(host_address, port)
    #print(URL)
    return redirect(URL)

@app.route('/ml_api_json', methods=['GET', 'POST'])
def ml_json():
    try:
        if request.method == 'POST':   
            #print('POST')
            try:
                model_key = request.form['model'] 
            except:
                model_key = None
                pass # Choice of models not implemented
            model_input_data = request.form['data']   
            
            output = score_routine(model_input_data, model_key=model_key, ETL=ETL, return_type='dict')
            
            if record_limit==1:
                return send_response(output[0])
            else:
                return send_response(output[0:record_limit])
        if request.method == 'GET': 
            #print('GET')
            try:
                model_key = request.args.get('model') 
            except:
                model_key = None
                pass # Choice of models not implemented
            model_input_data = request.args.get('data') 

            output = score_routine(model_input_data, model_key=model_key, ETL=ETL, return_type='dict')
            if record_limit==1:
                return send_response(output[0])
            else:
                return send_response(output[0:record_limit])
    except:
        print('API ERROR ! :\n{}\n'.format(traceback.format_exc()))
        return send_response('API ERROR ! Check input paramaters <br> [SERVER]/ml_api_json?model=<<modelkey>>&data={}')
    
    return '400: Bad request. Please check your request format.'

@app.route('/ml_api_file', methods=['GET','POST'])
def ml_file():
    if request.method == 'POST':
        print('POST')
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return '204: No file part'
        try:
           model_key = request.form['model'] 
        except:
            model_key = None
            pass # Choice of models not implemented
            
        file = request.files['file']
        print(file.filename)
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return '204: No selected file'
        
        if file and is_accepted_file_type(file.filename):            
            file_name = secure_filename(file.filename)
            file_path = os.path.join(temp_folder, file_name)
            file.save(file_path)

            model_input_data = pd.read_csv(file_path)            
            
            output = score_routine(model_input_data, model_key=model_key, ETL=ETL, return_type='frame')
            
            os.remove(file_path)
            #print(output)
#            stream = io.StringIO()
#            output.to_csv(stream, index=False, quoting=csv.QUOTE_ALL)
#            print(pd.read_csv(stream))
            #return output
            # return redirect(url_for('upload_file', filename=filename))
            #return send_response(output)
            #response = sendfile(stream, mimetype='text/csv')
#            response.headers['Content-Disposition'] = 'attachment; filename=data.csv'
#            print(response)
            ResultsFileName = 'output_{}.csv'.format(str(uuid.uuid4()))
            output.to_csv(os.path.join(static_folder, ResultsFileName), index=False, quoting=csv.QUOTE_ALL)
            response = send_from_directory(static_folder, ResultsFileName, as_attachment=True)
            os.remove(os.path.join(static_folder, ResultsFileName))
            return response
#            return send_file(stream, mimetype='text/csv', as_attachment=True, attachment_filename='output.csv')#, add_etags=True, cache_timeout=None, conditional=False, last_modified=None)

    return '400: Bad request. Please check your request format.'

@app.route('/input_form', methods=['GET','POST'])
def input_form():   
    
    ModelOptions = ''
    
    for mkey in MLModelObjects:
        ModelOptions = ModelOptions + Markup('<option value="{}">{}</option>\n'.format(mkey,mkey))
   
    if request.method == 'POST':   
        print('POST')
        #print(request.form['model'])
        try:
            model_key = request.form['model'] 
            if model_key=='None':
                model_key = None
        except:
            model_key = None
            pass # Choice of models not implemented
            
        if request.form['form']=='json':
            model_input_data = request.form['data']            
        elif request.form['form']=='file':
            if 'file' not in request.files:
                flash('No file part')
                return '204: No file part'
            file = request.files['file']
            #print(file.filename)
            # if user does not select file, browser also
            # submit a empty part without filename
            if file.filename == '':
                flash('No selected file')
                return '204: No selected file'
            
            if file and is_accepted_file_type(file.filename):            
                file_name = secure_filename(file.filename)
                file_path = os.path.join(temp_folder, file_name) #CHECK THIS
                file.save(file_path)    
                model_input_data = pd.read_csv(file_path)              
                os.remove(file_path)
                
        if request.form['form']=='file':            
            output, model_key = score_routine(model_input_data, model_key=model_key, ETL=ETL, return_type='frame', return_model_key=True)
        else:
            output, model_key = score_routine(model_input_data, model_key=model_key, ETL=ETL, return_type='dict', return_model_key=True)

        try:
            score_label = MLModelObjects[model_key].get_score_label()
            score_variable = MLModelObjects[model_key].get_score_variable()
                
            if request.form['form']=='json':
                #index_value = list(output.keys())[0]
                index_value = 0
                InputFileName = ''
                ResultsFileName = ''
                Actual = ''
                PredictionScore = output[index_value][score_label]
                PredictionProbability = np.round(output[index_value][score_variable],3)
                JSONOutput = json.dumps(output, indent=4, cls=custom_json_encoder)
            elif request.form['form']=='file':
                #index_value = None
                InputFileName = file_name
                ResultsFileName = 'output_{}.csv'.format(str(uuid.uuid4()))
                output.to_csv(os.path.join(static_folder,ResultsFileName), index=False, quoting=csv.QUOTE_ALL)
                Actual = ''
                PredictionScore = ''            
                PredictionProbability = ''
                JSONOutput = ''
        except:
                index_value = 0
                InputFileName = ''
                ResultsFileName = ''
                Actual = ''
                PredictionScore = ''            
                PredictionProbability = ''
                JSONOutput = json.dumps(output, indent=4, cls=custom_json_encoder)
        
        #print(JSONOutput)
        #JSONOutput.replace('\n', '&#13;&#10;')
        #print(JSONOutput)        
        return render_template("input_form.html", InputFileName=InputFileName, ResultsFileName=ResultsFileName, Actual=Actual, 
                               PredictionScore=PredictionScore, PredictionProbability=PredictionProbability,
                               JSONOutput=JSONOutput, ModelOptions=ModelOptions)

    return render_template("input_form.html", InputFileName='', ResultsFileName='', Actual='', PredictionScore='', 
                                   PredictionProbability='', JSONOutput='', ModelOptions=ModelOptions)

@app.route('/reload_models', methods=['GET','POST'])
def reload_models(): 
    # Setup key
    try:
        if request.method == 'POST':   
            #print('POST')
            acess_key = request.form['key'] 
        if request.method == 'GET': 
            #print('GET')
            acess_key = request.args.get('key') 
    except:
        acess_key = None
        print('API ERROR ! :\n{}\n'.format(traceback.format_exc()))
        return send_response('API ERROR ! Model Reload Failed !')
    
    # Reload
    try:
        if acess_key==config_access_key:
            init()
            return send_response('Models Chest Reloaded:'+str(MLModelObjects))
        else:
            return send_response('UNAUTHORIZED API ACCESS ! Model Reload Failed !')
    except:
        print('API ERROR ! :\n{}\n'.format(traceback.format_exc()))
        return send_response('API ERROR ! Model Reload Failed !')    

###############################################################################   
def init():
    global MLModelObjects
    global default_model_key
    global ETL
    global PostProcess
    
    try:
        model_chest_file_path = os.path.join(model_folder,'ModelsChest.json')
        with open(model_chest_file_path, 'r') as model_chest_file:
            model_chest_json = json.load(model_chest_file)
    except FileNotFoundError:
        print("Model Chest JSON File {} not found.".format(model_chest_file_path)) 
    
    model_chest_path = model_chest_json['Folder']
    
    if model_chest_path[0]=='*':        
        model_chest_path = os.path.join(model_folder, model_chest_path[1:])
    
    MLModelObjects = {}
    for mkey in model_chest_json['Models'].keys():
        model_object_file = os.path.join(model_chest_path, model_chest_json['Models'][mkey])
        MLModelObjects[mkey] = mltk.load_model(model_object_file)
        print('[Model {}]: {} loded...'.format(mkey, model_object_file))
    
    default_model_key = model_chest_json['Default']
    
    # LOAD ETL AND POST PROCESS FUNCTIONS
    ###############################################################################
    # ETL FUNCTION TO PROCESS DATA FOR SCORING
    try:
        import importlib.util
        spec1 = importlib.util.spec_from_file_location("etl_module",etl_py_script)
        etl_module = importlib.util.module_from_spec(spec1)
        spec1.loader.exec_module(etl_module)
        ETL = etl_module.ETL
    except:
        ETL = lambda DataFrame, variables_setup_dict, etl_parameters : DataFrame
        print('ERROR: \n{}'.format(traceback.format_exc()))
        print('No ETL tasks executed...\nInput datset will be not modified...')
    ###############################################################################    
    # POST PROCESS FUNCTION TO TAKE FINAL ACTION ON THE SCORED DATASET
    try:
        import importlib.util
        spec2 = importlib.util.spec_from_file_location("post_process_module",post_process_py_script)
        post_process_module = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(post_process_module)
        PostProcess = post_process_module.PostProcess
    except:
        PostProcess = lambda DataFrame, score_result_columns, file_reference, dropbox_folder, temp_folder, post_process_parameters : DataFrame
        print('ERROR: \n{}'.format(traceback.format_exc()))
        print('No PostProcess tasks executed...\nScored datset will be not modified...')
   ############################################################################### 
    
    # LOAD PRE-TRAINED MODEL
    #MLModelObjects[0] = mltk.load_model(ModelFile)
    #MLModelObject = MLModelObjects['LGR1'] 
    model_test_run(model_input_data_json, model_input_data_file)
       
###############################################################################
# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":
    #app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.json_encoder = custom_json_encoder
    app.config['DropBox'] = dropbox_folder
    app.config['TempFolder'] = temp_folder 
    app.config['SampleFileFolder'] = sample_folder
    app.config['JSON_SORT_KEYS'] = False
    app.config['APIKey'] = api_access_key
    app.config['CongigKey'] = config_access_key
    print((" * Loading ML Model and sttrating Flask API server. Please wait few momments to  until the server is fully started."))
    init()
    app.run(host=host_address, port=port, threaded=True, debug=False)
###############################################################################

