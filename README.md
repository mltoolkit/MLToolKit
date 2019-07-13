# MLToolkit 
## Current release: PyMLToolkit [v0.1.3]

<img src="MLToolkit.png" width="400">

MLToolkit (mltk) is a Python package providing a set of user-friendly functions to help building machine learning models in data science research, teaching or production focused projects. 

<img src="MLToolkit/image/MLTKProcess.png" height="200">

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
- Model Building
- Hyper Parameter Tuning [in development for v0.2]
- Model Performance Analysis and Comparison Between Models
- Auto ML (automated machine learning) [in development for v0.2]
- Model Deploymet and Serving [will be imporved for v0.2]

## Supported Machine Learning Algorithms/Packages
- RandomForestClassifier: scikit-learn
- LogisticRegression: statsmodels
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

## MLToolkit Example

### Data Loading and exploration
```python
import numpy as np
import pandas as pd
import mltk as mltk

Data = pd.read_csv(r'C:\Projects\Data\incomedata.csv')
Data = mltk.clean_column_names(Data, replace='') # Not available in v0.1.3
Data = mltk.add_identity_column(Data, id_label='ID', start=1, increment=1)
DataStats = mltk.data_description(Data)
```
### Data Pre-processing
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

categoryVariables = set({'sex', 'native-country', 'race', 'occupation', 'workclass', 'marital-status', 'relationship'})
binaryVariables = set({})
print(mltk.category_lists(Data, list(categoryVariables)))
```
```python
sourceVariable='age'
table = mltk.histogram(Data, sourceVariable, n_bins=10, orientation='vertical', show_plot=True)
print(table)

# Divide to categories
labels = ['0', '20', '30', '40', '50', '60', 'INF']
Data, groupVariable = mltk.numeric_to_category(DataFrame=Data, variable=sourceVariable, str_labels=labels, right_inclusive=True, print_output=False, return_variable=True)
mltk.plot_variable_response(DataFrame=Data, variable=groupVariable, class_variable=targetVariable)
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
TOTAL        32561        7841        100.00000          100.00000            NaN
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

model_parameters = {'MLAlgorithm':'RF', # 'LGR', #  'DFF', # 'CNN', # 'CATBST', # 'XGBST'
                    'NTrees':500,
                   'MaxDepth':100,
                   'MinSamplesToSplit':10,
                   'Processors':2} 

RFModel = mltk.build_ml_model(TrainDataset, ValidateDataset, TestDataset, model_variables, targetVariable, 
                 model_attributes, sample_attributes, model_parameters, score_parameters, 
                          return_model_object=True, show_results=False, show_plot=True)

print(RFModel.model_attributes['ModelID'])
print(RFModel.model_interpretation['ModelSummary'])
print(RFModel.model_evaluation['AUC'])
print(RFModel.model_evaluation['RobustnessTable'])

mltk.save_object(RFModel, '{}.pkl'.format(RFModel.get_model_id()))
# Save model
RFModel.plot_eval_matrics(comparison=False)
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
```python
TestDataset = mltk.score_dataset(TestDataset, RFModel, edges=None, score_label=None, fill_missing=0)
score_variable = RFModel.get_score_variable()
score_label = RFModel.get_score_label()

Robustnesstable = mltk.robustness_table(ResultsSet=TestDataset, class_variable=targetVariable, score_variable=score_variable,  score_label=score_label, show_plot=True)

threshold = 0.8
TestDataset['Predicted'] = np.where(TestDataset[score_variable]>threshold,1,0)
ConfusionMatrix = mltk.confusion_matrix(actual_variable=TestDataset[targetVariable], predcted_variable=TestDataset['Predicted'], labels=[0,1], sample_weight=None, totals=True)
print(ConfusionMatrix)

ConfusionMatrixRow = mltk.confusion_matrix_to_row(ConfusionMatrix, ModelID=RFModel.model_attributes['ModelID'])
ConfusionMatrixRow 
```
### Deployment
Simplified MLToolkit ETL pipeline for scoring and model re-building. (Need to customoze based on the project)
<img src="MLToolkit/image/MLTKServing.png" height="400">
```python
def ETL(DataFrame):
    # Add ID column
    DataFrame = mltk.add_identity_column(DataFrame, id_label='ID', start=1, increment=1)
    
    # Clean column names # Not avaiable in v0.1.3
    DataFrame = mltk.clean_column_names(DataFrame, replace='')
    input_columns = list(DataFrame.columns)
	
    # Excising valiables to model
    targetVariable = 'HighIncome'
    categoryVariables = set({'sex', 'nativecountry', 'race', 'occupation', 'workclass', 'maritalstatus', 'relationship'}) & set(DataFrame.columns)
    binaryVariables = set({}) & set(DataFrame.columns)
    
    # Check if target variable exists (fill the column with None in scoring)
    if not targetVariable in DataFrame.columns:
        DataFrame[targetVariable]=None
    
    # Create more Binary variables
    conditions = [{'bin_variable':'CapitalGainPositive', 'str_condition':"capitalgain>0"},
                  {'bin_variable':'CapitalLossPositive', 'str_condition':"capitalloss>0"}
                  ]
    DataFrame, bin_variables_ = mltk.create_binary_variables_set(DataFrame, conditions, return_variable=True) # Not available in v0.1.3
    binaryVariables.update(bin_variables_)
                  
    # Create more Catergorical variables
    buckets = [{'variable':'age', 'str_labels':['0', '20', '30', '40', '50', '60', 'INF']},
               {'variable':'educationnum', 'str_labels':['1', '4', '6', '8', '10', '13', '16']},
               {'variable':'hoursperweek', 'str_labels':['0', '20', '40', '50', '60', '80', 'INF']}
              ]    
    DataFrame, category_variables_ = mltk.create_categorical_variables_set(DataFrame, buckets, return_variable=True) # Not available in v0.1.3
    categoryVariables.update(category_variables_)
        
    #Finalize variables lists
    identifierColumns = ['ID']
    categoryVariables=list(categoryVariables)
    binaryVariables=list(binaryVariables)
    targetVariable = targetVariable

    # Create One Hot Encoded Variables
    DataFrame, featureVariables, targetVariable = mltk.to_one_hot_encode(DataFrame, category_variables=categoryVariables, binary_variables=binaryVariables, target_variable=targetVariable)
    
    return DataFrame, input_columns
```
Scoring
```python
MLModelObject = mltk.load_object(r'INCOMELEVELLGR20190704224750.pkl')
SampleDataset = pd.read_csv(r'test.csv')
SampleDataset = ETL(SampleDataset)

SampleDataset = mltk.score_dataset(SampleDataset, MLModelObject, edges=None, score_label=None, fill_missing=0)

Robustnesstable1 = mltk.robustness_table(ResultsSet=SampleDataset, class_variable=targetVariable, score_variable=score_variable,  score_label=score_label, show_plot=True)
```
### JSON Input for scoring

Records Format for single or fewer number if records
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
},]
```

Split Format for mulltiple records
```json
{
	"columns":["ID","age","education","hoursperweek","nativecountry","occupation","sex","workclass"],
	"data":[["A001",32,"Doctorate",40,"USA","Prof-specialty","Female","Private"],]
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
- 2018-07-02 [v0.0.1]: Initial set of functions for data exploration, model building and model evaluation was published to Github (https://github.com/sptennak/MachineLearning).
- 2019-03-20 [v0.1.0]: Developed and published initial version of model building and serving framework for IBM Coursera Advanced Data Science Capstone Project (https://github.com/sptennak/IBM-Coursera-Advanced-Data-Science-Capstone).
- 2019-07-02 [v0.1.2]: First release of the PyMLToolkit Python package, a collection of clases and functions facilitating end-to-end machine learning model building and serving over RESTful API.
- 2019-07-04 [v0.1.3]: Minor bug fixes.

## Future Release Plan
- 2019-07-14 [v0.1.4]: Enhancements and Minor bug fixes
- 2019-12-31 [v0.1.6]: Major bug-fix version of the initial release with some enhancements.
- [v0.2.0]: Imporved model serving frameework, support more machine learning algorithms and deep learning.
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
