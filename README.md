# MLToolkit 
## Current release: PyMLToolkit [v0.1.5]

<img src="https://raw.githubusercontent.com/sptennak/MLToolkit/master/MLToolkit.png" height="200">

MLToolkit (mltk) is a Python package providing a set of user-friendly functions to help building machine learning models in data science research, teaching or production focused projects. 

<img src="https://raw.githubusercontent.com/sptennak/MLToolkit/master/MLToolkit/image/MLTKProcess.png" height="200">

## Introduction
MLToolkit supports all stages of the machine learning application development process.

## Installation
```
pip install pymltoolkit
```
If the installation failed with dependancy issues, execute the above command with --no-dependencies

```
pip install pymltoolkit --no-dependencies
```

## Functions
- Data Extraction (SQL, Flatfiles, etc.)
- Exploratory Data Analysis (statistical summary, univariate analysis, etc.)
- Feature Engineering
- Model Building (Currently supported for binary classification only)
- Hyper Parameter Tuning [in development for v0.2]
- Model Performance Analysis and Comparison Between Models
- Model Building UI [in development for v0.2]
- ML Model Building Project [in development for v0.2]
- Auto ML (automated machine learning) [in development for v0.2]
- Model Deploymet and Serving [included, will be imporved for v0.2]

## Supported Machine Learning Algorithms/Packages
- RandomForestClassifier: scikit-learn
- LogisticRegression: statsmodels
- Deep Feed Forward Neural Network (DFF): tensorflow
- Convlutional Neural Network (CNN): tensorflow
- Gradient Boost : catboost
- ... More models will be added in the future releases ...

## Usage
```python
import mltk
```

### Warning: Python Variable, Function or Class names 
The Python interpreter has a number of built-in functions. It is possible to overwrite thier definitions when coding without any rasing a warning from the Python interpriter. (https://docs.python.org/3/library/functions.html)
Therfore, AVOID THESE NAMES as your variable, function or class names.
<table border="1">
<tr><td>abs</td><td>all</td><td>any</td><td>ascii</td><td>bin</td><td>bool</td><td>bytearray</td><td>bytes</td></tr>
<tr><td>callable</td><td>chr</td><td>classmethod</td><td>compile</td><td>complex</td><td>delattr</td><td>dict</td><td>dir</td></tr>
<tr><td>divmod</td><td>enumerate</td><td>eval</td><td>exec</td><td>filter</td><td>float</td><td>format</td><td>frozenset</td></tr>
<tr><td>getattr</td><td>globals</td><td>hasattr</td><td>hash</td><td>help</td><td>hex</td><td>id</td><td>input</td></tr>
<tr><td>int</td><td>isinstance</td><td>issubclass</td><td>iter</td><td>len</td><td>list</td><td>locals</td><td>map</td></tr>
<tr><td>max</td><td>memoryview</td><td>min</td><td>next</td><td>object</td><td>oct</td><td>open</td><td>ord</td></tr>
<tr><td>pow</td><td>print</td><td>property</td><td>range</td><td>repr</td><td>reversed</td><td>round</td><td>set</td></tr>
<tr><td>setattr</td><td>slice</td><td>sorted</td><td>staticmethod</td><td>str</td><td>sum</td><td>super</td><td>tuple</td></tr>
<tr><td>type</td><td>vars</td><td>zip</td><td>__import__</td></tr>
</table>

If you accedently overwrite any of the built-in function (e.g. list), execute the following to bring built-in defition.
```python
del(list)
```

Similarly, avoid using special charcters and spaces in the column names of the DataFrames.
Execute the following to remove special characters from the column names.
```python
Data = mltk.clean_column_names(Data, replace='')
```

## MLToolkit Example

### Data Loading and exploration
```python
import numpy as np
import pandas as pd
import mltk as mltk

Data = mltk.read_data_csv(file=r'C:\Projects\Data\incomedata.csv')
Data = mltk.clean_column_names(Data, replace='')
Data = mltk.add_identity_column(Data, id_label='ID', start=1, increment=1)
DataStats = mltk.data_description(Data)
```
### Data Pre-processing and Feature Engineering
```python
# Analyze Response Target
print(mltk.variable_frequency(DataFrame=Data, variable='income'))

# Set Target Variables
targetVariable = 'HighIncome'
targetCondition = "income=='>50K'" #For Binary Classification

Data=mltk.set_binary_target(Data, target_condition=targetCondition, target_variable=targetVariable)
print(mltk.variable_frequency(DataFrame=Data, variable=targetVariable))
```
```
        Counts  CountsFraction%
income                         
<=50K    24720         75.91904
>50K      7841         24.08096
TOTAL    32561        100.00000
```
```python
# Flag Records to Exclude
excludeCondition="age < 18"
action = 'flag' # 'drop' #
excludeLabel = 'EXCLUDE'
Data=mltk.exclude_records(Data, exclude_ondition=excludeCondition, action=action, exclude_label=excludeLabel) # )#

# Get list of uniques values in categorical variables
categoryVariables = set({'sex', 'nativecountry', 'race', 'occupation', 'workclass', 'maritalstatus', 'relationship'})
print(mltk.category_lists(Data, list(categoryVariables)))

# Merge unique categorical values
category_merges = [{'variable':'maritalstatus', 'category_variable':'maritalstatus', 'group_value':'Married', 'values':["Married-civ-spouse", "Married-spouse-absent", "Married-AF-spouse"]}]
Data = mltk.merge_categories(Data, category_merges)

# Show Frequency distribution of categorical variable
sourceVariable='maritalstatus'
table = mltk.variable_frequency(Data, variable=sourceVariable, show_plot=False)
table.style.background_gradient(cmap='Greens').set_precision(3)

# Response Rate For Categorical Variables
mltk.variable_responses(Data, variables=categoryVariables, target_variable=targetVariable, show_output=False, show_plot=True)
```

```python
# Transform numeric variable
transformations = [{ "variable":"age", "to": "normalizedage", "operation":"normalize", "parameters":{"method":"zscore"}}]
Data, transformed_variables = mltk.create_transformed_variables(Data, variable_operations=transformations, return_variables=True)

# Create Categorical Variables from continious variables
sourceVariable='age'
table = mltk.histogram(Data, sourceVariable, n_bins=10, orientation='vertical', show_plot=True)
print(table)

# Divide to categories
buckets = [{'variable':sourceVariable, 'category_variable':None,'str_labels':['0', '20', '30', '40', '50', '60', 'INF'], 'right_inclusive':True}]
Data, categoryVariable = mltk.create_categorical_variables(Data, buckets, return_variables=True)
mltk.plot_variable_response(DataFrame=Data, variable=categoryVariable, class_variable=targetVariable)
```
```
            Counts  HighIncome  CountsFraction%  ResponseFraction%  ResponseRate%
ageGRP                                                                           
1_(0,20]      2410           2          7.40149            0.02551        0.08299
2_(20,30]     8162         680         25.06680            8.67236        8.33129
3_(30,40]     8546        2406         26.24612           30.68486       28.15352
4_(40,50]     6983        2655         21.44590           33.86048       38.02091
5_(50,60]     4128        1547         12.67774           19.72963       37.47578
6_(60,INF)    2332         551          7.16194            7.02716       23.62779
TOTAL        32561        7841        100.00000          100.00000        0.24081
```
```python
# Divide to categories using JSON input
create_variable_task_dict = """
{
	"type":"buckets",
	"variable_class": "cat",
	"include":true,
	"show_output":false,
	"rule_set":[
		{
			"variable":"hoursperweek", 
			"category_variable":null, 
			"str_labels":["0", "20", "40", "50", "60", "80", "INF"]
		}
	]
}
"""
Data, category_variable = create_variable_task(Data, create_variable_task_dict=create_variable_task_dict)

```

```python
# Create One Hot Encoded Variables
Data, featureVariables, targetVariable = mltk.to_one_hot_encode(Data, category_variables=categoryVariables, binary_variables=binaryVariables, target_variable=targetVariable)
Data[identifierColumns+featureVariables+[targetVariable]].sample(5).transpose()
```
### Correlation
```python
correlation=mltk.correlation_matrix(Data, featureVariables+[targetVariable], target_variable=targetVariable, method='pearson', return_type='list', show_plot=False)
```
### Split Train, Validate Test datasets
```python
TrainDataset, ValidateDataset, TestDataset = mltk.train_validate_test_split(Data, ratios=(0.6,0.2,0.2))
```
### Model Building
```python
sample_attributes = {'SampleDescription':'Adult Census Income Dataset',
                    'NumClasses':2,
                    'RecordIdentifiers':identifierColumns
                    }

score_parameters = {'Edges':[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                   'Quantiles':10,
                   'ScoreVariable':'Probability',
                   'ScoreLabel':'Score',
                   'QuantileLabel':'Quantile'
                   }

model_attributes = {'ModelID': None,   
                   'ModelName': 'IncomeLevel',
                   'Version':'0.1',
                   }
```

Losgistic Regression
```python
model_parameters = {'MLAlgorithm':'LGR', # 'RF', # 'DFF', # 'CNN', # 'CATBST', # 'XGBST'
                    'MaxIterations':50}  
```

Random Forest
```python
model_parameters = {'MLAlgorithm':'RF', # 'LGR', #  'DFF', # 'CNN', # 'CATBST', # 'XGBST'
                    'NTrees':500,
                   'MaxDepth':100,
                   'MinSamplesToSplit':10,
                   'Processors':2} 
```
Neural Networks
```python
# Setup Architecture
# Binary classification (L1 'units': 2), 32 variables ('input_shape':(48,))
SimpleDFF_architecture = {
        'L1':{'type': 'Dense', 'position':'input', 'units': 512, 'activation':'relu', 'input_shape':(48,)},
        'L2':{'type': 'Dense', 'position':'hidden', 'units': 512, 'activation':'relu'},
        'L3':{'type': 'Dropout', 'position':'hidden', 'rate':0.5},
        'L4':{'type': 'Dense', 'position':'output', 'units': 2, 'activation':'softmax', 'output_shape':None},
       }

# Binary classification (L1 'units': 2), 32 variables ('input_shape':(32,))
LogisticRegressionNN_architecture = {
        'L1':{'type': 'Dense', 'position':'input', 'units': 2, 'activation':'softmax', 'input_shape':(32,)},
       }

# Binary classification (L8 'units': 2)
SimpleImageClassifier_architecture = {
        'L1':{'type': 'Conv2D', 'position':'input', 'filters': 32, 'kernel_size':(3,3), 'strides':(1,1), 'padding':'valid', 'activation':'relu', 'input_shape':(128, 128, 1)},
        'L2':{'type': 'Conv2D', 'position':'hidden', 'filters': 64, 'kernel_size':(3,3), 'strides':(1,1), 'padding':'valid', 'activation':'relu'},
        'L3':{'type': 'MaxPooling2D', 'position':'hidden', 'pool_size': (2,2), 'padding':'valid'},   
        'L4':{'type': 'Dropout', 'position':'hidden', 'rate':0.25},
        'L5':{'type': 'Flatten', 'position':'hidden'},        
        'L6':{'type': 'Dense', 'position':'hidden', 'units': 128, 'activation':'relu'},
        'L7':{'type': 'Dropout', 'position':'hidden', 'rate':0.5},
        'L8':{'type': 'Dense', 'position':'output', 'units': 2, 'activation':'softmax', 'output_shape':None},
       }
	   
model_parameters = {'MLAlgorithm':'NN',
                    'BatchSize':512,
                   'InputShape':InputShape,
                   'num_classes':2,
                   'Epochs':10,
                   'metrics':['accuracy'],
                   'architecture':SimpleDFF_architecture} 
```
CatBoost
```python
model_parameters = {'MLAlgorithm':'CBST',
                    'NTrees': 500,
                    'MaxDepth':10,
                    'LearningRate':0.7,
                    'LossFunction':'Logloss',#crossEntropy
                    'EvalMatrics':'Accuracy',
                    'Imbalanced':False,
                    'TaskType':'GPU',
                    'Processors':2,
                    'UseBestModel':True}
```

### Build Model
```python
XModel = mltk.build_ml_model(TrainDataset, ValidateDataset, TestDataset, 
                                  model_variables=modelVariables,
                                  variable_setup = None,
                                  target_variable=targetVariable,
                                  model_attributes=model_attributes, 
                                  sample_attributes=sample_attributes, 
                                  model_parameters=model_parameters, 
                                  score_parameters=score_parameters, 
                                  return_model_object=True, 
                                  show_results=False, 
                                  show_plot=True
                                  )

print(XModel.model_attributes['ModelID'])
print(XModel.model_interpretation['ModelSummary'])
print('ROC AUC: ', XModel.get_auc(curve='roc'))
print('PRC AUC: ', XModel.get_auc(curve='prc'))
print(XModel.model_evaluation['RobustnessTable'])

XModel.plot_eval_matrics(comparison=False)
```

```
          minProbability  maxProbability  meanProbability  BucketCount  ResponseCount  BucketFraction  ResponseFraction  BucketPrecision  CumulativeBucketFraction  CumulativeResponseFraction  CumulativePrecision
Quantile                                                                                                                                                                                                           
1                0.00000         0.00008      3.85729e-06          652            3         0.10011           0.00192          0.00460                   1.00000                     1.00000              0.23967
2                0.00008         0.00432      1.52655e-03          651            9         0.09995           0.00577          0.01382                   0.89989                     0.99808              0.26582
3                0.00435         0.02042      1.10941e-02          652           14         0.10011           0.00897          0.02147                   0.79994                     0.99231              0.29731
4                0.02049         0.05702      3.58648e-02          650           20         0.09980           0.01281          0.03077                   0.69983                     0.98334              0.33677
5                0.05711         0.12075      8.51409e-02          652           65         0.10011           0.04164          0.09969                   0.60003                     0.97053              0.38767
6                0.12086         0.20457      1.63366e-01          651          109         0.09995           0.06983          0.16743                   0.49992                     0.92889              0.44533
7                0.20469         0.31870      2.61577e-01          651          190         0.09995           0.12172          0.29186                   0.39997                     0.85906              0.51478
8                0.31895         0.46840      4.03550e-01          666          259         0.10226           0.16592          0.38889                   0.30002                     0.73735              0.58905
9                0.46854         0.66965      5.68083e-01          641          377         0.09842           0.24151          0.58814                   0.19776                     0.57143              0.69255
10               0.66994         0.99967      8.06834e-01          647          515         0.09934           0.32992          0.79598                   0.09934                     0.32992              0.79598
DataSet          0.00000         0.99967      2.33167e-01         6513         1561         1.00000           1.00000          0.23967                   1.00000                     1.00000              0.23967
```

### Evaluate Model

Plot model performance curves
```python
RFModel.plot_eval_matrics(comparison=True)
LGRModel.plot_eval_matrics(comparison=True)
NNModel.plot_eval_matrics(comparison=True)
CBSTModel.plot_eval_matrics(comparison=True)
```

Area Under Curve (AUC) Comparison
```python
Models = [LGRModel, RFModel, CBSTModel, NNModel]
ModelsComp = mltk.model_guages_comparison(Models)
print(ModelsComp)
```

```
                           Model  PRC_AUC  ROC_AUC
0   INCOMELEVELLGR20190728113633  0.71971  0.88926
1    INCOMELEVELRF20190728113635  0.69348  0.88113
2  INCOMELEVELCBST20190728113703  0.71507  0.88975
3    INCOMELEVELNN20190728113641  0.71396  0.88890
```

Test Model
```python
score_variable = RFModel.get_score_variable()
score_label = RFModel.get_score_label()

TestDataset = mltk.score_processed_dataset(TestDataset, RFModel, edges=None, score_label=None, fill_missing=0)

threshold = 0.8
TestDataset = mltk.set_predicted_columns(TestDataset, score_variable, threshold=threshold)
ConfusionMatrix = mltk.confusion_matrix(TestDataset, actual_variable=targetVariable, predcted_variable='Predicted', labels=[0,1], sample_weight=None, totals=True)
print(ConfusionMatrix)
```

Comparing Models and Probability Thresholds
```python
Models = [LGRModel, RFModel, CBSTModel, NNModel]
thresholds=[0.7, 0.8, 0.9]
ConfusionMatrixComparison = mltk.confusion_matrix_comparison(TestDataset, Models, thresholds, score_variable=None, show_plot=True)
ConfusionMatrixComparison.style.background_gradient(cmap='RdYlGn').set_precision(3)
```

Comparing Models and Threshold Score (1-10 Scale)
```python
Models = [LGRModel, RFModel, CBSTModel, NNModel]
thresholds=[7, 8, 9]
ConfusionMatrixComparison = mltk.confusion_matrix_comparison(TestDataset, Models, thresholds, score_variable=score_label, show_plot=True)
ConfusionMatrixComparison.style.background_gradient(cmap='RdYlGn').set_precision(3)
```

Set Custom Score Edges
``` python
RobustnessTable, ROCCurve, PrecisionRecallCurve, roc_auc, prc_auc = mltk.model_performance_matrics(ResultsSet=TestDataset, target_variable=targetVariable, score_variable=score_variable, quantile_label='Quantile',  quantiles=100, show_plot=True)
print('ROC AUC', roc_auc)
print('PRC AUC', prc_auc)

print(RobustnessTable)

# Re-bin score buckets
edges = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.95, 1.0]
LGRModel.set_score_edges(edges)
```

Save model
```python
saveFilePath = '{}.pkl'.format(XModel.get_model_id())
mltk.save_model(XModel, saveFilePath)
```

### Deployment
Simplified MLToolkit ETL pipeline for scoring and model re-building (Need to customize based on the project).
<img src="https://raw.githubusercontent.com/sptennak/MLToolkit/master/MLToolkit/image/MLTKServing.png" height="300">

Define ETL Function
```python
def ETL(DataFrame):
    # Add ID column
    DataFrame = mltk.add_identity_column(DataFrame, id_label='ID', start=1, increment=1)
    
    # Clean column names
    DataFrame = mltk.clean_column_names(DataFrame, replace='')
    input_columns = list(DataFrame.columns)

    variables_setup_dict = """   
    {
        "setting":"model",

        "variables": {            
                "category_variables" : ["sex", "nativecountry", "race", "occupation", "workclass", "maritalstatus", "relationship"],
                "binary_variables": [],
                "target_variable":"HighIncome"
        },


        "transform": {
            "type":"transform",
            "variable_class": "num",
            "include":false,
            "show_output":false,
            "rule_set": [
                {
                    "variable":"age", 
                    "to": "normalizedage", 
                    "operation":"normalize", 
                    "parameters":{"method":"zscore"}
                },
                {
                    "variable":"hoursperweek", 
                    "to": "normalizedhoursperweek", 
                    "operation":"normalize", 
                    "parameters":{"method":"minmaxfs"}
                }
            ]
        },

        "conditions":{
            "type":"conditions",
            "variable_class": "bin",
            "include":true,
            "show_output":false,
            "rule_set":[        
                {
                    "bin_variable":"CapitalGainPositive", 
                    "str_condition":"capitalgain>0"
                },

                {
                    "bin_variable":"CapitalLossPositive", 
                    "str_condition":"capitalloss>0"
                }
            ]
        },

        "buckets":{
            "type":"buckets",
            "variable_class": "cat",
            "include":true,
            "show_output":false,
            "rule_set":[
                {
                    "variable":"age", 
                    "category_variable":null, 
                    "str_labels":["0", "20", "30", "40", "50", "60", "INF"]
                },
                {
                    "variable":"educationnum", 
                    "category_variable":null, 
                    "str_labels":["1", "4", "6", "8", "10", "13", "16"]
                },
                {
                    "variable":"hoursperweek", 
                    "category_variable":null, 
                    "str_labels":["0", "20", "40", "50", "60", "80", "INF"]
                }
            ]
        },

        "category_merges":{
            "type":"category_merges",
            "variable_class": "cat",
            "include":true,
            "show_output":false,
            "rule_set":[
                {
                    "variable":"maritalstatus", 
                    "category_variable":"maritalstatus", 
                    "group_value":"Married", 
                    "values":["Married-civ-spouse", "Married-spouse-absent", "Married-AF-spouse"]
                }
            ]
        },

        "pair_equality": {
            "type":"pair_equality",
            "variable_class": "cat",
            "include":false,
            "show_output":false,
            "rule_set":[
                {
                    "variable1":"age", 
                    "variable2":"hoursperweek",
                    "category_variable":"TestGroup"
                }
            ]
        }
    }
    """    
    
    DataFrame, categoryVariables, binaryVariables, targetVariable = mltk.setup_variables_task(DataFrame, variables_setup_dict)
    
    # Create One Hot Encoded Variables
    DataFrame, featureVariables, targetVariable = mltk.to_one_hot_encode(DataFrame, category_variables=categoryVariables, binary_variables=binaryVariables, target_variable=targetVariable)

    return DataFrame, input_columns
```

Scoring
```python
MLModelObject = mltk.load_model(saveFilePath)
SampleDataset = pd.read_csv(r'test.csv')
SampleDataset = ETL(SampleDataset)

SampleDataset = mltk.score_processed_dataset(SampleDataset, MLModelObject, edges=None, score_label=None, fill_missing=0)
Robustnesstable1 = mltk.robustness_table(ResultsSet=SampleDataset, class_variable=targetVariable, score_variable=score_variable,  score_label=score_label, show_plot=True)
```

```python
MLModelObject = mltk.load_model(saveFilePath)

TestInput = """
{
      "ID": "A001",
      "age": 32,
      "workclass": "Private",
      "education": "Doctorate",
      "education-num": 16,
      "marital-status": "Married-civ-spouse",
      "occupation": "Prof-specialty",
      "relationship": "Husband",
      "race": "Asian-Pac-Islander",
      "sex": "Male",
      "capital-gain": 0,
      "capital-loss": 0,
      "hours-per-week": 40,
      "native-country": "?"
}
"""
output = mltk.score_records(TestInput, MLModelObject, edges=None, ETL=ETL, return_type='dict') # Other options for return_type, {'json', 'frame'}
```
Output
```python
[{'ID': 'A001',
 'age': 32,
 'capitalgain': 0,
 'capitalloss': 0,
 'education': 'Doctorate',
 'educationnum': 16,
 'hoursperweek': 40,
 'maritalstatus': 'Married',
 'nativecountry': '?',
 'occupation': 'Prof-specialty',
 'race': 'Asian-Pac-Islander',
 'relationship': 'Husband',
 'sex': 'Male',
 'workclass': 'Private',
 'Probability': 0.6790258814478549,
 'Score': 7}]
```
### JSON Input for scoring

Records Format for single or fewer number of records
```json
[{
	"ID": "A001",
	"age": 32,
	"workclass": "Private",
	"education": "Doctorate",
	"occupation": "Prof-specialty",
	"sex": "Female",
	"hoursperweek": 40,
	"nativecountry": "USA"
}]
```

Split Format for mulltiple records
```json
{
	"columns":["ID","age","education","hoursperweek","nativecountry","occupation","sex","workclass"],
	"data":[["A001",32,"Doctorate",40,"USA","Prof-specialty","Female","Private"]]
}
```

## License
```
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
```
## MLToolkit Project Timeline
- 2018-07-02 [v0.0.1]: Initial set of functions for data exploration, model building and model evaluation was published to Github. (https://github.com/sptennak/MachineLearning).
- 2018-01-03 [v0.0.2]: Created more functions for data exploration including web scraping and geo spacial data analysis for for IBM Coursera Data Science Capstone Project was published to Github. (https://github.com/sptennak/Coursera_Capstone).
- 2019-03-20 [v0.1.0]: Developed and published initial version of model building and serving framework for IBM Coursera Advanced Data Science Professional Certificate Capstone Project. (https://github.com/sptennak/IBM-Coursera-Advanced-Data-Science-Capstone).
- 2019-07-02 [v0.1.2]: First release of the PyMLToolkit Python package, a collection of clases and functions facilitating end-to-end machine learning model building and serving over RESTful API.
- 2019-07-04 [v0.1.3]: Minor bug fixes.
- 2019-07-14 [v0.1.4]: Improved documentation, Integrated TensorFlow Models, Enhancements and Minor bug fixes.
- 2019-07-28 [v0.1.5]: Integrated CatBoost Models, Improved model building and serving frameework, support JSON input/output to the ML model bulding and scoring processes, Enhancements and bug fixes.

## Future Release Plan
- TBD [v0.1.6]: Integrate image classification model Deployment, UI Preview, ML Model Building Projects, Enhancements and bug fixes.
- 2019-12-31 [v0.1.7]: Comprehensive documentation, Post implementation evaluation functions, Major bug-fix version of the initial release with finalized enhancements.
- [v0.2.0]: Imporved model building and serving frameework, UI, support more machine learning algorithms and deep learning.
- [v0.3.0]: Imporved scalability and performance, Hyper parameter tuning, and Automated Machine Learning.
- [v0.4.0]: Building continious learning models.

## References
- https://pandas.pydata.org/
- https://scikit-learn.org
- https://www.numpy.org/
- https://docs.python.org/3.6/library/re.html
- https://www.statsmodels.org
- https://matplotlib.org/
- http://flask.pocoo.org/
- https://catboost.ai/
- http://json.org/
