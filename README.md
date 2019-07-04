# MLToolkit 
## Current release: PyMLToolkit [v0.1.3]

<img src="MLToolkit.png" width="400">

MLToolkit (mltk) is a Python package providing a set of user-friendly functions to help building machine learning models in data science research, teaching or production focused projects. 

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

## Examples
### Data Loading and exploration
```python
import numpy as np
import pandas as pd
import mltk as mltk

Data = pd.read_csv(r'C:\Projects\Data\incomedata.csv')
Data = mltk.add_identity_column(Data, id_label='ID', start=1, increment=1)
DataStats = mltk.data_description(Data)
```
### Data Preprocessing
```python
# Analyze Response Target
print(mltk.variable_frequency(DataFrame=Data, variable='income'))

# Set Target Valriable
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
1                0.00000         0.00406          0.00064         1306           11.0         0.20052           0.00705          0.00842                   1.00000                     1.00000              0.23967
2                0.00406         0.02182          0.01138          648           15.0         0.09949           0.00961          0.02315                   0.79948                     0.99295              0.29768
3                0.02184         0.05599          0.03639          651           34.0         0.09995           0.02178          0.05223                   0.69998                     0.98334              0.33670
4                0.05603         0.11695          0.08490          652           64.0         0.10011           0.04100          0.09816                   0.60003                     0.96156              0.38408
5                0.11731         0.20303          0.15683          651          109.0         0.09995           0.06983          0.16743                   0.49992                     0.92056              0.44134
6                0.20323         0.31633          0.26482          651          182.0         0.09995           0.11659          0.27957                   0.39997                     0.85074              0.50979
7                0.31654         0.48098          0.39805          653          249.0         0.10026           0.15951          0.38132                   0.30002                     0.73414              0.58649
8                0.48136         0.67088          0.57195          651          380.0         0.09995           0.24343          0.58372                   0.19975                     0.57463              0.68947
9                0.67233         1.00000          0.80734          650          517.0         0.09980           0.33120          0.79538                   0.09980                     0.33120              0.79538
DataSet          0.00000         1.00000          0.23319         6513         1561.0         1.00000           1.00000          0.23967                   1.00000                     1.00000              0.23967
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
- 2019-07-02 [v0.1.2]: First resease of the PyMLToolkit Python package, a collection of clases and functions facilitating end-to-end machine learning model building and serving over RESTful API.
- 2019-07-04 [v0.1.3]: Minor bug fix

## Future Resease Plan
- 2019-12-31 [v0.1.6]: Major bug-fix version of the initial resease.
- [v0.2.0]: Imporved model serving frameework, support more machine learning algorithms and deep learning.
- [v0.3.0]: Hyper parameter tuning and Automated machine learning.
- [v0.4.0]: Building continious learning models.

## References
- https://pandas.pydata.org/
- https://scikit-learn.org
- https://www.numpy.org/
- https://docs.python.org/3.6/library/re.html
- https://www.statsmodels.org
- https://matplotlib.org/
- http://flask.pocoo.org/
