# MLToolkit 
## Current release: PyMLToolkit [v0.1.2]

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
```
import mltk
```

## Examples
### Data Loading and exploration
```python
import numpy as np
import pandas as pd
import mltk as mltk

Data = pd.read_pickle('Modeling_Dataset.pkl')

Data = mltk.add_identity_column(Data, id_label='ID', start=1, increment=1)

DataStats = mltk.data_description(Data)
targetVariable='VendorAcceptedAcceleratedCashOffer'

table = mltk.histogram(Data, 'age', n_bins=10, orientation='vertical')
print(table)

table = mltk.histogram(Data, 'hours-per-week', n_bins=[0, 10, 20, 40,60, 80, np.inf], orientation='vertical')
print(table)
```
### Data Preprocessing
```python
variable='age'
labels = ['0', '20', '30', '40', '50', '60', 'INF']
Data, groupVariable = mltk.numeric_to_category(DataFrame=Data, variable=sourceVariable, str_labels=labels, left_inclusive=True, print_output=False, return_variable=True)
mltk.plot_variable_response(DataFrame=Data, variable=groupVariable, class_variable=targetVariable)

mltk.plot_variable_responses(Data, variables=categoryVariables+binaryVariables, class_variable=targetVariable)
```
### Correlation
```python
correlation=mltk.correlation_matrix(Data, featureVariables+[targetVariable], target_variable=targetVariable, method='pearson', return_type='list', show_plot=False)
print(correlation.head())
```
### Split Train, Validate Test datasets
```python
TrainDataset, ValidateDataset, TestDataset = mltk.train_validate_test_split(Data, ratios=(0.6,0.2,0.2))
```
### Model Building
```python
#... Example code in preperation ....#
````
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

## Future Resease Plan
- 2019-12-31 [v1.6.0]: Major bug-fix version of the initial resease.
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
