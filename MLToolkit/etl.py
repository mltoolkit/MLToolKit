# -*- coding: utf-8 -*-
# MLToolkit (mltoolkit)

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

# IF QUERY HAS DROP TABLE, TRUNCATE TABLE, DELETE, UPDATE, CREATE, set a control flag on (safety)
# Modify output timing to execute query + row count

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
import urllib
import sqlalchemy 
import csv

try:
    import pyodbc
except:
    print('pyodbc not found! Data base query fufnctions disabled.')
import warnings
warnings.filterwarnings("ignore")

from mltk.string import *
from mltk.explore import *

def number_unit_example():
    edges_std = ['0', '1p', '1n', '1u', '1m', '1c', '1', '100', '500', 
                 '1K', '2K', '5K', '10K', '20K', '50K', '100K', '500K', 
                 '1M', '2M', '5M', '10M', '100M', '200M', '500M', 
                 '1G', '2G', '5G', '10G', '100G', '200G', '500G',
                 '1T', '2T', '5T', '10T', '100T', '200T', '500T',
                 '1P', '2P', '5P', '10P', '100P', '200P', '500P',
                 '1E']
    print(edges_std)
    
def get_number_units():    
    units = {'p':0.000000000001,
        'n':0.000000001,
        'u':0.000001,
        'm':0.001,
        'c':0.01,
        'd':0.1,
        '':1,
        'D':10,
        'H':100,
        'K':1000,
        'M':1000000,
        'G':1000000000,
        'T':1000000000000,
        'P':1000000000000000,
        'E':1000000000000000000,
        'INF':np.inf        
        }
    units = pd.DataFrame(data=units.items(), columns=['unit', 'multiplier'])
    print(units)
    return units

###############################################################################
##[ I/O FUNCTIONS]#############################################################      
###############################################################################
def read_data(connector, params=None):
    connector = {
            "method":"sql", #"pickle", "csv", "excel"
            "source":{"dbms":"mssql", "server":"SQLSERVER1", "database":"SampleDB", "schema":None},
            "auth":{'type':'user', 'user':'user1', 'pwd':'password123'},
            "params":{}
            }
    
    connector2 = {
            "method":"sql", #"pickle", "csv", "excel"
            "source":{"dbms":"snowflake", "server":"SQLSERVER1", "database":"SampleDB", "schema":None}, # account (server)
            "auth":{'type':'user', 'user':'user1', 'password':'password123', "role": None}, 
            "params":{}
            }
    
    return None    
    
def read_data_csv(file, separator=',', quoting= 'MINIMAL', compression='infer', encoding='utf-8'):
    """
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
    
    Parameters
    ----------    
    file : str
    separator : str
    index : bool
    compression : {'infer', 'gzip', 'bz2', 'zip', 'xz', None}, default 'infer'
    quoting : {'ALL', MINIMAL', 'NONNUMERIC', 'NONE'}, default 'MINIMAL'
    encoding : {'utf-8', 'utf-16'}, default 'utf-8'
     
    Returns
    -------
    DataFrame : pandas.DataFrame
    """
    if quoting=='ALL':
        quoting = csv.QUOTE_ALL
    elif quoting=='MINIMAL':
        quoting = csv.QUOTE_MINIMAL        
    elif quoting=='NONNUMERIC':
        quoting = csv.QUOTE_NONNUMERIC        
    elif quoting=='NONE':
        quoting = csv.QUOTE_NONE   
    
    try:
        start_time = timer() 
        DataFrame = pd.read_csv(filepath_or_buffer=file, sep=separator, quoting=quoting, 
                                compression=compression, encoding=encoding)  
        execute_time = timer() - start_time
    except:
        execute_time = 0
        DataFrame = pd.DataFrame()
        print(traceback.format_exc())
        
    
    
    print('{:,d} records were loaded. execute time = {} s'.format(len(DataFrame.index), execute_time))
    
    return DataFrame

def write_data_csv(DataFrame, file, separator=',', index=False, quoting='ALL', encoding='utf-8', compression='infer', chunksize=None):
    """
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
    
    Parameters
    ----------    
    DataFrame : pandas.DataFrame
    file : str
    separator : str
    index : bool
    compression : {'infer', 'gzip', 'bz2', 'zip', 'xz', None}, default 'infer'
    quoting : {'ALL', MINIMAL', 'NONNUMERIC', 'NONE'}, default 'MINIMAL'
    encoding : {'utf-8', 'utf-16'}, default 'utf-8'
    chunksize : int, default None
    
    Returns
    -------
    None
    """
    
    if quoting=='ALL':
        quoting = csv.QUOTE_ALL
    elif quoting=='MINIMAL':
        quoting = csv.QUOTE_MINIMAL        
    elif quoting=='NONNUMERIC':
        quoting = csv.QUOTE_NONNUMERIC        
    elif quoting=='NONE':
        quoting = csv.QUOTE_NONE        
    try:
        start_time = timer()     
        DataFrame.to_csv(path_or_buf=file, sep=separator, encoding=encoding, index=index, 
                         quoting=quoting, compression=compression, chunksize=chunksize)
        execute_time = timer() - start_time
    except:
        execute_time = 0
        print(traceback.format_exc())
        
    print('{:,d} records were written. execute time = {} s'.format(len(DataFrame.index), execute_time))
    
    return None

def read_data_pickle(file, compression='infer'):
    """
    https://docs.python.org/3/library/pickle.html
    "Warning The pickle module is not secure against erroneous or maliciously constructed data. 
    Never unpickle data received from an untrusted or unauthenticated source."
    
    Parameters
    ----------    
    file : str
    compression : {'infer', 'gzip', 'bz2', 'zip', 'xz', None}, default 'infer'
    
    Returns
    -------
    DataFrame : pandas.DataFrame
    """
    try:
        start_time = timer() 
        DataFrame = pd.read_pickle(path=file, compression=compression)
        execute_time = timer() - start_time
    except:
        execute_time = 0
        print(traceback.format_exc())
        DataFrame = pd.DataFrame()
        
        
    print('{:,d} records were loaded. execute time = {} s'.format(len(DataFrame.index), execute_time))
    
    return DataFrame

def write_data_pickle(DataFrame, file, compression='infer', protocol=3):
    """
    https://docs.python.org/3/library/pickle.html
    "Warning The pickle module is not secure against erroneous or maliciously constructed data. 
    Never unpickle data received from an untrusted or unauthenticated source."
    
    Parameters
    ----------    
    DataFrame : pandas.DataFrame
    file : str
    compression : {'infer', 'gzip', 'bz2', 'zip', 'xz', None}, default 'infer'
    protocol : int {1, 2, 3, 4}
        0 is human-readable/backwards compatible with earlier versions of Python
        read more at https://docs.python.org/3/library/pickle.html
    Returns
    -------
    None    
    """
    try:
        start_time = timer() 
        DataFrame.to_pickle(path=file, compression=compression, protocol=protocol)
        execute_time = timer() - start_time
    except:
        execute_time = 0
        print(traceback.format_exc())
        
    print('{:,d} records were written. execute time = {} s'.format(len(DataFrame.index), execute_time))

def create_sql_connect_string(server=None, database=None, auth=None, dbms='mssql', autocommit = 'True'):    
    if dbms=='mssql':
        # Download ODBC Driver https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server
        driver = 'ODBC Driver 13 for SQL Server' # 'SQL Server' # 
        if auth['type']=='machine':
            connect_string = r'Driver={'+driver+'};SERVER='+server+';DATABASE='+database+';TRUSTED_CONNECTION=yes;autocommit='+autocommit+';'
            connect_string = urllib.parse.quote_plus(connect_string)
        elif auth['type']=='user':
            uid =  auth['uid'] 
            pwd =  auth['pwd'] 
            connect_string = r'Driver={'+driver+'};SERVER='+server+';DATABASE='+database+';UID='+uid+'r;PWD='+pwd+'; autocommit='+autocommit+';'
            connect_string = urllib.parse.quote_plus(connect_string)
    elif dbms=='mysql':
        connect_string = None
    elif dbms=='snowflake':
        connect_string = None
    else:
        raise Exception("Parameter dbms not provided. Accepted values are {'mssql', 'mysql', 'snowflake'}")
    return connect_string

def read_data_sql(query=None, server=None, database=None, auth=None, dbms='mssql', params=None):
    """
    Parameters
    ----------
    query : str
        SQL SELECT query
    server : str
        Database Server
    database : str
        Database
    auth :  dict
        e.g. auth = {'type':'user', 'uid':'user', 'pwd':'password'} for username password authentication
             auth = {'type':'machine', 'uid':None, 'pwd':None} for machine authentication
    
    Returns
    -------
    DataFrame : pandas.DataFrame
    """   
    execute_time = 0
    
    if query!=None and server!=None and auth!=None:        
        coerce_float=True
        index_col=None
        parse_dates=None
        
        try:
            if auth['type']=='machine':
                connect_string = r'Driver={SQL Server};SERVER='+server+';DATABASE='+database+';TRUSTED_CONNECTION=yes;'
            elif auth['type']=='user':
                uid =  auth['uid'] 
                pwd =  auth['pwd'] 
                connect_string = r'Driver={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+uid+'r;PWD='+pwd+'}'
            else:
                raise Exception('No db server authentication method provided!')
            connection = pyodbc.connect(connect_string)   
            
            start_time = timer() 
            DataFrame = pd.read_sql_query(sql=query, con=connection, coerce_float=coerce_float, index_col=index_col, parse_dates=parse_dates)
            execute_time = timer() - start_time
            
            connection.close()        
        except:
            print('Database Query Fialed!:\n{}\n'.format(traceback.format_exc()))
            DataFrame=pd.DataFrame()
    else:
        print('No Query provided !')
        DataFrame=pd.DataFrame()
        
    print('{:,d} records were loaded. execute time = {} s'.format(len(DataFrame.index), execute_time))
   
    return DataFrame

def write_data_sql(DataFrame, server=None, database=None, schema=None, table=None, index=False, dtypes=None, if_exists='fail', auth=None, dbms='mssql', params=None):
    """
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_sql.html
    
    Parameters
    ----------
    DataFrame : pandas.DataFrame
        DataFrame
    server : str
        Database Server
    database : str
        Database
    schema : str
        Database Schema
    table : str
        Table name
    if_exists : {'fail', 'replace', 'append'}, default 'fail'
        Action if the table already exists.
    auth :  dict
        e.g. auth = {'type':'user', 'uid':'user', 'pwd':'password'} for username password authentication
             auth = {'type':'machine', 'uid':None, 'pwd':None} for machine authentication
    
    Returns
    -------
    None
    """ 
    
    # Download ODBC Driver https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server
    driver = 'ODBC Driver 13 for SQL Server' # 'SQL Server' # 
    autocommit = 'True'
    fast_executemany = True
    execute_time = 0
    
    if server!=None and  database!=None and schema!=None and table!=None and auth!=None : 
        try:
            if auth['type']=='machine':
                #connect_string = r'Driver={SQL Server};SERVER='+server+';DATABASE='+database+';TRUSTED_CONNECTION=yes;' #ODBC (slow)
                connect_string = r'Driver={'+driver+'};SERVER='+server+';DATABASE='+database+';TRUSTED_CONNECTION=yes;autocommit='+autocommit+';'
                connect_string = urllib.parse.quote_plus(connect_string)
            elif auth['type']=='user':
                uid =  auth['uid'] 
                pwd =  auth['pwd'] 
                #connect_string = r'Driver={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+uid+'r;PWD='+pwd+'}' #ODBC (slow)
                connect_string = r'Driver={'+driver+'};SERVER='+server+';DATABASE='+database+';UID='+uid+'r;PWD='+pwd+'; autocommit='+autocommit+';'
                connect_string = urllib.parse.quote_plus(connect_string)
            else:
                raise Exception('No db server authentication method provided !') 
                
            #connection = pyodbc.connect(connect_string) #ODBC (slow)
            engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect="+connect_string, fast_executemany=fast_executemany)
            connection = engine
            
            start_time = timer() 
            if dtypes==None:
                DataFrame.to_sql(name=table, con=connection, schema=schema, index= index, if_exists=if_exists)
            else:
                DataFrame.to_sql(name=table, con=connection, schema=schema, index= index, dtype=dtypes, if_exists=if_exists)
            execute_time = timer() - start_time
            
            #connection.close() 
            engine.dispose()
            rowcount = len(DataFrame.index)
        except:
            print('Database Query Failed! Check If ODBC driver installed. \nIf not, Download ODBC Driver from https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-serve.:\n{}\n'.format(traceback.format_exc()))
            rowcount = 0
    else:
        print('Check the destiniation table path (server, database, schema, table, auth) !')
        rowcount = 0
    
    print('{:,d} records were written. execute time = {} s'.format(rowcount, execute_time))
    
    return rowcount

def execute_sql_query(query=None, server=None, database=None, auth=None, params=None, dbms='mssql', on_error='ignore'):
    """
    Parameters
    ----------
    query : str
        SQL SELECT query
    server : str
        Database Server
    database : str
        Database
    auth :  dict
        e.g. auth = {'type':'user', 'uid':'user', 'pwd':'password'} for username password authentication
             auth = {'type':'machine', 'uid':None, 'pwd':None} for machine authentication
    params : dict
        extra parameters (not implemented)
        
    Returns
    -------
    DataFrame : pandas.DataFrame
    """        
    # Download ODBC Driver https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server
    driver = 'ODBC Driver 13 for SQL Server' # 'SQL Server' # 
    autocommit = 'True'
    fast_executemany = True
    
    if server!=None and  database!=None and query!=None and auth!=None :
        try:
            if auth['type']=='machine':
                connect_string = r'Driver={'+driver+'};SERVER='+server+';DATABASE='+database+';TRUSTED_CONNECTION=yes;autocommit='+autocommit+';'
                connect_string = urllib.parse.quote_plus(connect_string)
                
            elif auth['type']=='user':
                uid =  auth['uid'] 
                pwd =  auth['pwd'] 
                connect_string = r'Driver={'+driver+'};SERVER='+server+';DATABASE='+database+';UID='+uid+'r;PWD='+pwd+'; autocommit='+autocommit+';'
                connect_string = urllib.parse.quote_plus(connect_string)
            else:
                raise Exception('No db server authentication method provided !')
            
            engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect="+connect_string, fast_executemany=fast_executemany)
            
            # connection
            connection = engine.connect()
            
            #transaction
            trans = connection.begin()
        
            # execute
            start_time = timer() 
            result = connection.execute(query)
            execute_time = timer() - start_time
            
            try:
                rowcount = result.rowcount
                print('{} rows affected. execute time = {} s'.format(rowcount,execute_time))
            except:
                rowcount = -1
                print('ERROR in fetching affected rows count. execute time = {} s'.format(execute_time))
                
            # commit
            trans.commit()
        
            # close connections, results set and dispose engine (moved to finally)
            #connection.close()
            #result.close()
            #engine.dispose()
        except:
            print(r'ERROR: Check If ODBC driver installed. \nIf not, Download ODBC Driver from https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server:\n{}\n'.format(traceback.format_exc()))
            rowcount = 0
        finally:
            # close connections, results set and dispose engine
            try:
                connection.close()
            except:
                print('Failed to close connection !')
            try:
                result.close()
            except:
                print('Failed to close results !')
            try:
                engine.dispose()
            except:
                print('Failed to dispose engine !')
            
        return rowcount      

def sql_server_database_list(server, auth=None, user_database_only=True, dbms='mssql'):
    """
    Reference: https://docs.microsoft.com/en-us/sql/relational-databases/system-compatibility-views/sys-sysdatabases-transact-sql?view=sql-server-2017
    """
    
    query = """
    SELECT 
        @@SERVERNAME AS [ServerName],
        NAME AS [DBName],
        STATUS AS [Status],
        CRDATE AS [CreateDate]
    FROM master.dbo.sysdatabases (NOLOCK)
    WHERE Name NOT IN ( 'master','tempdb','model' ,'msdb')
    """    
    DBList = read_data_sql(query=query, server=server, database='master', auth=auth, params=None)
    
    return DBList
    
def sql_server_database_usage_report(server, database, auth=None, schema=None, table=None, user_tables_only=True, dbms='mssql', unit='KB'):
    """
    Reference: https://docs.microsoft.com/en-us/sql/relational-databases/system-catalog-views/sys-tables-transact-sql?view=sql-server-2017
    """
    
    if user_tables_only:
        user_tables_only_condition = "AND table.is_ms_shipped = 0 " # is_ms_shipped = 1 (indicates this object was shipped or created by Microsoft), 0 (indicates this object was created by a user)
    else:
        user_tables_only_condition = ""
        
    if schema != None:
        schema_condition = "AND schema.NAME = '{}'".format(schema)
    else:
        schema_condition = ""

    if table != None:
        table_condition = "AND table.NAME = '{}'".format(table)
    else:
        table_condition = ""

    #Unit conversion
    if unit == 'KB':
        multiplier = 1.0
    if unit  == 'MB':
        multiplier = 1.0/1024.0
    if unit  == 'GB':
        multiplier = 1.0/(1024.0*1024.0)  
    if unit  == 'TB':
        multiplier = 1.0/(1024.0*1024.0*1024.0) 
    
    if dbms == 'mssql':
        query = """
        SELECT
            @@SERVERNAME AS [Server],
            DB_Name() AS [DB],
            [schema].NAME AS [Schema],
            [table].NAME AS [Table],
            [table].CREATE_DATE AS [CreateDate],
            [table].MODIFY_DATE AS [ModifyDate],		
            [part].ROWS AS [Rows],
            SUM(alloc.total_pages) * 8 AS [TotalSpaceKBx],
            SUM(alloc.used_pages) * 8 AS [UsedSpaceKBx],
        FROM
            sys.tables [table] (NOLOCK)
        INNER JOIN     
            sys.indexes (NOLOCK) [ix] ON ([table].OBJECT_ID = [ix].OBJECT_ID)
        INNER JOIN
            sys.partitions (NOLOCK) [part] ON ([ix].OBJECT_ID = [part].OBJECT_ID AND ix.index_id = [part].index_id)
        INNER JOIN
            sys.allocation_units (NOLOCK) [alloc] ON ([part].PARTITION_ID = [alloc].container_id)
        LEFT OUTER JOIN
            sys.schemas [schema] (NOLOCK) ON ([table].SCHEMA_ID = [schema].SCHEMA_ID)
        WHERE
            [table].NAME IS NOT NULL
            {user_tables_only_condition}
            {table_condition}
            {schema_condition}
        GROUP BY
            [table].NAME, 
            [table].CREATE_DATE, 
            [table].MODIFY_DATE, 
            [schema].NAME, part.ROWS
        """.format(schema_condition=schema_condition, table_condition=table_condition, user_tables_only_condition=user_tables_only_condition)
        
        DBUsageReport = read_data_sql(query=query, server=server, database=database, auth=auth, params=None)
        
        DBUsageReport['TotalSpaceKBx'] = DBUsageReport['TotalSpaceKBx'].fillna(0)
        DBUsageReport['UsedSpaceKBx'] = DBUsageReport['UsedSpaceKBx'].fillna(0)
        DBUsageReport['AvaiableSpaceKBx'] = DBUsageReport['TotalSpaceKBx'] - DBUsageReport['UsedSpaceKBx']
        
        DBUsageReport['TotalSpace{}'.format(unit)] = DBUsageReport['TotalSpaceKBx'] * multiplier
        DBUsageReport['UsedSpace{}'.format(unit)] = DBUsageReport['UsedSpaceKBx'] * multiplier
        DBUsageReport['AvaiableSpace{}'.format(unit)] = DBUsageReport['AvaiableSpaceKBx'] * multiplier
        
        DBUsageReport = DBUsageReport.drop(columns=['TotalSpaceKBx', 'UsedSpaceKBx', 'AvaiableSpaceKBx'])
    else:
        DBUsageReport = pd.DataFrame()
        print('This function currently supported for MSSQL server only')
    
    return DBUsageReport

###############################################################################
##[ VALIDATE FIELDS]##########################################################      
###############################################################################
        
def add_identity_column(DataFrame, id_label='ID', start=1, increment=1):
    if id_label in DataFrame.columns:
        print('Column {} exists in the DataFrame'.format(id_label))
        return DataFrame
    else:
        DataFrame.reset_index(drop=True, inplace=True)
        DataFrame.insert(0, id_label, start+DataFrame.index)
        return DataFrame
    
def remove_special_characters(str_val, replace=''):
    return re.sub('\W+',replace, str_val)

def remove_special_characters_list(str_list, replace=''):
    return [remove_special_characters(str_val, replace=replace) for str_val in str_list]
    
def clean_column_names(DataFrame, replace=''): # Remove special charcters from column names
    """
    Parameters
    ----------
    DataFrame : pandas.DataFrame
        DataFrame
    replace : str, dafault ''
        Character to replace special charaters with.    
    
    Returns
    -------
    DataFrame : pandas.DataFrame
    """
    try:
        columns = DataFrame.columns
        columns = remove_special_characters_list(columns, replace=replace)
        if check_list_values_unique(columns):
            DataFrame.columns = columns
        else:
            print('Duplicates values excists the column names after removing special characters!. Column names were rolled-back to initial values.')        
    except:
        print('Error in removing special characters from column names:\n{}\n'.format(traceback.format_exc()))
    return DataFrame

def check_list_values_unique(values_list):
    if len(values_list) == len(set(values_list)):
        return True
    else:
        return False
    
def handle_duplicate_columns(DataFrame, action='rename'): #'drop'
    """
    Parameters
    ----------
    DataFrame : pandas.DataFrame
        DataFrame
    action : {'rename', 'drop'}, dafault 'rename'
        Action to be taken on duplicate columns    
    
    Returns
    -------
    DataFrame : pandas.DataFrame
    """
    is_duplicate = DataFrame.columns.duplicated()
    columns = list(DataFrame.columns)
    if action=='rename':
        for i in range(len(columns)):
            if is_duplicate[i]:
               columns[i]=columns[i]+'_' 
        DataFrame.columns = columns
    elif action=='drop':
        DataFrame = DataFrame.loc[:,~is_duplicate]
    else:
        print('No valid action (rename or drop) provided!')
    return DataFrame

def add_missing_feature_columns(DataFrame, expected_features, fill_value=0):
    # Blanck columns for non-existance variables
    feature_variables_to_add = list(set(expected_features) - set(DataFrame.columns)) # Find columns not found in the dataset
    for f in feature_variables_to_add:
        DataFrame[f]=fill_value
        print('Column [{}] does not exist in the dataset. Created new column and set to {}...'.format(f,fill_value))
    return DataFrame

def exclude_records(DataFrame, exclude_condition=None, action = 'flag', exclude_label='_EXCLUDE_'):
    N0 = len(DataFrame.index)
    if exclude_condition==None:
        print('No exclude condition...')
        return DataFrame
    
    try:
        if action=='drop': #Drop Excludes        
            DataFrame = DataFrame.query('not ({})'.format(exclude_condition))
        elif action=='flag': #Create new flagged column
            DataFrame[exclude_label] = DataFrame.eval(exclude_condition).astype('int8')
            print('Records {} -> {}=1'.format(exclude_condition, exclude_label))
    except:
        print('Error in excluding records {}:\n{}\n'.format(exclude_condition, traceback.format_exc()))
    N1 = len(DataFrame.index)    
    print('{} records were excluded'.format(N1-N0))
    return DataFrame

###############################################################################
##[ CREATING FEATURES - TARGET ]###############################################      
############################################################################### 
    
def set_binary_target(DataFrame, to_variable='_TARGET_', condition_str=None, default=0, null=0, return_variable=False, return_script=False):
    if condition_str==None: 
        return DataFrame
    
    DataFrame, to_variable = create_binary_variable(DataFrame, to_variable, condition_str, default=default, null=null, return_variable=True)

    parameters = {
            'condition_str':condition_str,
            'default':default,
            'null':null
            }    
    script_dict = generate_create_variable_task_script(type='target', out_type='bin', include=False, operation='condition', source=None, destination=to_variable, parameters=parameters)
    
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame 
    
###############################################################################
##[ CREATING FEATURES - TRANSFORMATIONS]#######################################      
############################################################################### 

def create_normalized_variable(DataFrame, variable, method='maxscale', parameters=None, to_variable=None, return_variable=False, return_script=False):
    if to_variable==None:
        to_variable = variable
        
    if method=='minscale': #scale=max
        try:
            min_ = parameters["min"]
        except:
            min_ = DataFrame[variable].min()
            parameters["min"] = min_
        DataFrame[to_variable] = DataFrame[variable]/min_
    if method=='maxscale': #scale=max
        try:
            max_ = parameters["max"]
        except:
            max_ = DataFrame[variable].max()
            parameters["max"] = max_
        DataFrame[to_variable] = DataFrame[variable]/max_
    if method=='range': # range = abs(max-min)
        try:
            min_ = parameters["min"]
            max_ = parameters["max"]
        except:
            min_ = DataFrame[variable].min()
            max_ = DataFrame[variable].max()    
            parameters["min"] = min_
            parameters["max"] = max_
        min_max = abs(min_-max_)
        DataFrame[to_variable] = DataFrame[variable]/min_max
    if method=='minmaxfs': # range = (value-min)/(max-min)
        try:
            min_ = parameters["min"]
            max_ = parameters["max"]
        except:
            min_ = DataFrame[variable].min()
            max_ = DataFrame[variable].max() 
            parameters["min"] = min_
            parameters["max"] = max_
        min_max = abs(max_-min_)
        DataFrame[to_variable] = (DataFrame[variable]-min_)/min_max
    if method=='minmaxfs_m': # range = (value-min)/(max-min)
        try:
            min_ = parameters["min"]
            max_ = parameters["max"]
            mean_ = parameters["mean"]
        except:  
            min_=DataFrame[variable].min()
            max_=DataFrame[variable].max()
            mean_ = DataFrame[variable].mean()
            parameters["min"] = min_
            parameters["max"] = max_
            parameters["mean"] = mean_
        min_max = abs(max_-min_)
        DataFrame[to_variable] = (DataFrame[variable]-mean_)/min_max
    if method=='mean':
        try:
            mean_ = parameters["mean"]
        except:  
            mean_ = DataFrame[variable].mean()
            parameters["mean"] = mean_
        DataFrame[to_variable] = DataFrame[variable]/mean_
    if method=='median':
        try:
            median_ = parameters["median"]
        except:  
            median_ = DataFrame[variable].median()
            parameters["median"] = median_
        DataFrame[to_variable] = DataFrame[variable]/median_
    if method=='zscore':  
        try:
            std_ = parameters["std"]
            mean_ = parameters["mean"]
        except: 
            std_ = DataFrame[variable].std()
            mean_ = DataFrame[variable].mean()
            parameters["mean"] = mean_
            parameters["std"] = std_
        DataFrame[to_variable] = (DataFrame[variable] - mean_)/std_  
 
    script_dict = generate_create_variable_task_script(type='transform', out_type='cnt', 
                                                       include=False, operation='normalize', 
                                                       source=variable, destination=to_variable, 
                                                       parameters=parameters)
     
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame 

def create_datepart_variable(DataFrame, variable, to_variable=None, part='date', return_variable=False, return_script=False):
    if to_variable==None:
        to_variable = '{}{}'.format(variable,part)
        
    try:
        DataFrame[variable] = pd.to_datetime(DataFrame[variable])
        if part=='date':
            DataFrame[to_variable] = DataFrame[variable].dt.date
        elif part=='year':
            DataFrame[to_variable] = DataFrame[variable].dt.year
        elif part=='quarter':
            DataFrame[to_variable] = DataFrame[variable].dt.quarter
        elif part=='month':
            DataFrame[to_variable] = DataFrame[variable].dt.month
        elif part=='week':
            DataFrame[to_variable] = DataFrame[variable].dt.week
        elif part=='day':
            DataFrame[to_variable] = DataFrame[variable].dt.day  
        elif part=='dayofweek':
            DataFrame[to_variable] = DataFrame[variable].dt.dayofweek
        elif part=='dayofyear':
            DataFrame[to_variable] = DataFrame[variable].dt.dayofyear
        elif part=='time':
            DataFrame[to_variable] = DataFrame[variable].dt.time
        elif part=='hour':
            DataFrame[to_variable] = DataFrame[variable].dt.hour
        elif part=='minute':
            DataFrame[to_variable] = DataFrame[variable].dt.minute
        elif part=='second':
            DataFrame[to_variable] = DataFrame[variable].dt.second
        elif part=='microsecond':
            DataFrame[to_variable] = DataFrame[variable].dt.microsecond
        elif part=='nanosecond':
            DataFrame[to_variable] = DataFrame[variable].dt.nanosecond
        else:
            DataFrame[to_variable] = variable
    except:
        DataFrame[to_variable] = variable

    parameters = {'part':part}    
    script_dict = generate_create_variable_task_script(type='transform', out_type='dat', 
                                                       include=False, operation='datepart', 
                                                       source=variable, destination=to_variable, 
                                                       parameters=parameters)
   
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame 

def create_dateadd_variable(DataFrame, variable, to_variable=None, unit='years', value=0, return_variable=False, return_script=False):
    if to_variable==None:
        to_variable = '{}{}{}'.format(variable, value, unit)
        
    try:
        DataFrame[variable] = pd.to_datetime(DataFrame[variable])
        if part=='years':
            DataFrame[to_variable] = DataFrame[variable] + pd.DateOffset(year=value)
        elif part=='months':
            DataFrame[to_variable] = DataFrame[variable] + pd.DateOffset(months=value)
        elif part=='weeks':
            DataFrame[to_variable] = DataFrame[variable] + pd.DateOffset(weeks=value)
        elif part=='days':
            DataFrame[to_variable] = DataFrame[variable] + pd.DateOffset(days=value)
        elif part=='hours':
            DataFrame[to_variable] = DataFrame[variable] + pd.DateOffset(hours=value)
        elif part=='minutes':
            DataFrame[to_variable] = DataFrame[variable] + pd.DateOffset(minutes=value)
        elif part=='seconds':
            DataFrame[to_variable] = DataFrame[variable] + pd.DateOffset(seconds=value)
        elif part=='microseconds':
            DataFrame[to_variable] = DataFrame[variable] + pd.DateOffset(microseconds=value)
        elif part=='nanoseconds':
            DataFrame[to_variable] = DataFrame[variable] + pd.DateOffset(nanoseconds=value)        
    except:
        DataFrame[to_variable] = variable

    parameters = {
        'unit':unit, 
        'value':value
        }    
    script_dict = generate_create_variable_task_script(type='transform', out_type='dat', 
                                                       include=False, operation='dateadd', 
                                                       source=variable, destination=to_variable, 
                                                       parameters=parameters)
        
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame   

def create_log_variable(DataFrame, variable, base='e', to_variable=None, return_variable=False, return_script=False):
    if to_variable==None:
        to_variable = 'LOG{}'.format(variable)
        
    if base=='e':
        DataFrame[to_variable] = np.log(DataFrame[variable])
    elif base=='10':
        DataFrame[to_variable] = np.log10(DataFrame[variable])
    elif base=='2':
        DataFrame[to_variable] = np.log2(DataFrame[variable])

    parameters = { 'base':base }
    script_dict = generate_create_variable_task_script(type='transform', out_type='cnt', 
                                                       include=False, operation='log', 
                                                       source=variable, destination=to_variable, 
                                                       parameters=parameters) 
        
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame     
    
def create_exponent_variable(DataFrame, variable, base='e', to_variable=None, return_variable=False, return_script=False):
    if to_variable==None:
        to_variable = 'EXP{}'.format(variable)
        
    if base=='e':
        DataFrame[to_variable] = np.e**DataFrame[variable]
    elif base=='10':
        DataFrame[to_variable] = 10**DataFrame[variable]
    elif base=='2':
        DataFrame[to_variable] = 2**DataFrame[variable]

    parameters = { 'base':base }
    script_dict = generate_create_variable_task_script(type='transform', out_type='cnt', 
                                                       include=False, operation='exponent', 
                                                       source=variable, destination=to_variable, 
                                                       parameters=parameters) 
        
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame 


def create_segmented_variable(DataFrame, variable, a=None, b=None, to_variable=None, return_variable=False, return_script=False):
    if to_variable==None:
        to_variable = 'SEG{}'.format(variable)  
    
    if a == None:
        a = -np.inf
    
    if b == None:
        b = np.inf
        
    DataFrame[to_variable] = DataFrame[variable]
    DataFrame.loc[DataFrame[to_variable]<a, to_variable] = a
    DataFrame.loc[DataFrame[to_variable]>b, to_variable] = b

    parameters = { 'a':a, 'b':b }
    script_dict = generate_create_variable_task_script(type='transform', out_type='cnt', 
                                                       include=False, operation='segment', 
                                                       source=variable, destination=to_variable, 
                                                       parameters=parameters)     
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame         
###############################################################################
##[ CREATING FEATURES - STR TRANSFORM ]########################################      
############################################################################### 
        
def create_str_count_variable(DataFrame, variable, pattern='*', case_sensitive=True, to_variable=None, return_variable=False, return_script=False):
    if to_variable==None:
        to_variable = '{}CNT{}'.format(variable, remove_special_characters(pattern, replace=''))
    try:
        if pattern=='*':
            DataFrame[to_variable] = DataFrame[variable].str.len()
        else:
            DataFrame[to_variable] = DataFrame[variable].str.count(pattern) 
    except:
        print('ERROR in create_str_count_variable:\n{}'.format(traceback.format_exc()))
        DataFrame[to_variable] = DataFrame[variable]

    parameters = { 'pattern':pattern, 'case_sensitive':case_sensitive }
    script_dict = generate_create_variable_task_script(type='transform_str', out_type='cnt', 
                                                       include=False, operation='strcount', 
                                                       source=variable, destination=to_variable, 
                                                       parameters=parameters) 
            
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame 
 
def create_str_normalized_variable(DataFrame, variable, to_case='lower', chars='keep', numbers='remove', spchar='remove', space='remove', to_variable=None, return_variable=False, return_script=False):

    if to_variable==None:
        to_variable = '{}'.format(variable)
    
    try:    
        DataFrame[to_variable] = DataFrame[variable]
        
        if to_case=='lower':
            DataFrame[to_variable] = DataFrame[variable].str.lower()
        if to_case=='upper':
            DataFrame[to_variable] = DataFrame[variable].str.upper()
        if numbers=='remove':
            DataFrame[to_variable] = DataFrame[variable].str.replace('\d','')    
        if spchar=='remove':
            DataFrame[to_variable] = DataFrame[variable].str.replace('\W','')   
        if space=='remove':
            DataFrame[to_variable] = DataFrame[variable].str.replace('\s','')       
        if chars=='remove':
            DataFrame[to_variable] = DataFrame[variable].str.replace('\w','') 
    except:
        print('ERROR in create_str_normalized_variable:\n{}'.format(traceback.format_exc()))
        DataFrame[to_variable] = DataFrame[variable]

    parameters = { 
        'to_case':to_case, 
        'chars':chars,
        'numbers':numbers,
        'spchar':spchar, 
        'space':space
    }
    script_dict = generate_create_variable_task_script(type='transform_str', out_type='str', 
                                                       include=False, operation='normalize', 
                                                       source=variable, destination=to_variable, 
                                                       parameters=parameters) 
        
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame  

def create_str_extract_variable(DataFrame, variable, pattern='\w+', case_sensitive=True, to_variable=None, return_variable=False, return_script=False): 
    if to_variable==None:
        to_variable = 'variableEXT'.format(variable)
    try:
        if case_sensitive:    
            DataFrame[to_variable] = DataFrame[variable].str.extract('({})'.format(pattern))
        else:
            DataFrame[to_variable] = DataFrame[variable].str.extract('({})'.format(pattern), flags=re.IGNORECASE)
    except:
        print('ERROR in create_str_extract_variable:\n{}'.format(traceback.format_exc()))
        DataFrame[to_variable] = DataFrame[variable]

    parameters = { 
        'pattern':pattern, 
        'case_sensitive':case_sensitive
    }
    script_dict = generate_create_variable_task_script(type='transform_str', out_type='str', 
                                                       include=False, operation='extract', 
                                                       source=variable, destination=to_variable, 
                                                       parameters=parameters) 
        
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame 

###############################################################################
##[ CREATING FEATURES - MULTI VARIABLE ]#######################################      
###############################################################################    
def create_operation_mult_variable(DataFrame, expression_str='0', to_variable=None, return_variable=False, return_script=False):
    if to_variable==None:
        to_variable = '{}'.format(expression_str)
    
    try:
        DataFrame[to_variable] = DataFrame.eval(expression_str)
    except:
        print('ERROR in create_operation_mult_variable:\n{}'.format(traceback.format_exc()))

    parameters = { 'expression_str':expression_str}
    script_dict = generate_create_variable_task_script(type='operation_mult', out_type='cnt', 
                                                       include=False, operation='expression', 
                                                       source=None, destination=to_variable, 
                                                       parameters=parameters) 
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame  

###############################################################################
##[ CREATING FEATURES - SEQUENCE ORDER ]#######################################      
###############################################################################
def create_sequence_order_variable(DataFrame, variable1a, variable2a, variable1b, variable2b, output='binary', to_variable=None, return_variable=False, return_script=False):
    if to_variable==None:
        to_variable = '{}{}SEQ{}{}'.format(variable1a, variable2a, variable1b, variable2b)
        
    try:
        DataFrame[to_variable] = DataFrame[variable] ########### NEED UPDATE !!!!
    except:
        print('ERROR in create_sequence_order_variable:\n{}'.format(traceback.format_exc()))
        DataFrame[to_variable] = DataFrame[variable]

    parameters = { 'output':output }
    script_dict = generate_create_variable_task_script(type='sequence', out_type='cnt', 
                                                       include=False, operation='seqorder', 
                                                       source=[variable1a, variable2a, variable1b, variable2b], 
                                                       destination=to_variable, 
                                                       parameters=parameters) 
        
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame  
    
###############################################################################
##[ CREATING FEATURES - DIFFERENCES ]##########################################      
############################################################################### 
def create_numeric_difference_variable(DataFrame, variable1, variable2, multiplier=1, onerror=None, to_variable=None, return_variable=False, return_script=False):
    if to_variable==None:
        to_variable = '{}DIFF{}'.format(variable1, variable2)
    
    try:
        DataFrame[variable1] = pd.to_numeric(DataFrame[variable1], errors='coerce')
        DataFrame[variable2] = pd.to_numeric(DataFrame[variable2], errors='coerce')        
        DataFrame[to_variable] = multiplier*(DataFrame[variable1] - DataFrame[variable2])
    except:
        DataFrame[to_variable] = None
        print('Data Type Error in {}, {} : {} '.format(variable1, variable2, traceback.format_exc()))  

    parameters = { 
                    'multiplier':multiplier,
                    'onerror': onerror
    }
    script_dict = generate_create_variable_task_script(type='comparison', out_type='cnt', 
                                                       include=False, operation='numdiff', 
                                                       source=[variable1, variable2], 
                                                       destination=to_variable, 
                                                       parameters=parameters) 
    
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame 

def create_numeric_ratio_variable(DataFrame, variable1, variable2, multiplier=1, onerror=None, to_variable=None, return_variable=False, return_script=False):
    if to_variable==None:
        to_variable = '{}DIV{}'.format(variable1, variable2)
    
    try:
        DataFrame[variable1] = pd.to_numeric(DataFrame[variable1], errors='coerce')
        DataFrame[variable2] = pd.to_numeric(DataFrame[variable2], errors='coerce')        
        DataFrame[to_variable] = multiplier*(DataFrame[variable1]/DataFrame[variable2])
    except:
        DataFrame[to_variable] = None
        print('Data Type Error in {}, {} : {} '.format(variable1, variable2, traceback.format_exc()))  

    parameters = { 
                    'multiplier':multiplier,
                    'onerror': onerror
    }
    script_dict = generate_create_variable_task_script(type='comparison', out_type='cnt', 
                                                       include=False, operation='ratio', 
                                                       source=[variable1, variable2], 
                                                       destination=to_variable, 
                                                       parameters=parameters) 
    
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame 
    
def create_date_difference_variable(DataFrame, variable1, variable2, to_variable=None, unit='day', onerror=None, return_variable=False, return_script=False):
    if to_variable==None:
        to_variable = '{}DIFF{}'.format(variable1,variable2)
    
    try:
        DataFrame[variable1] = pd.to_datetime(DataFrame[variable1])
        DataFrame[variable2] = pd.to_datetime(DataFrame[variable2])        
        DataFrame[to_variable] = DataFrame[variable2] - DataFrame[variable1]
        DataFrame[to_variable]=DataFrame[to_variable]/np.timedelta64(1,unit)
    except:
        DataFrame[to_variable] = None
        print('Date Type Error in {}, {} : {} '.format(variable1, variable2, traceback.format_exc()))  

    parameters = { 
                    'unit':unit,
                    'onerror': onerror
    }
    script_dict = generate_create_variable_task_script(type='comparison', out_type='cnt', 
                                                       include=False, operation='datediff', 
                                                       source=[variable1, variable2], 
                                                       destination=to_variable, 
                                                       parameters=parameters) 
    
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame    

def create_row_min_variable(DataFrame, variable1, variable2, to_variable=None, return_variable=False, return_script=False):
    
    if to_variable==None:
        to_variable = '{}MIN{}'.format(variable1,variable2)
    
    try:
        DataFrame[to_variable] = DataFrame[[variable1,variable2]].min(axis=1)
    except:
        DataFrame[to_variable] = None
        print('Row min({}, {}) Error: {}'.format(variable1, variable2, traceback.format_exc()))  

    parameters = {  }
    script_dict = generate_create_variable_task_script(type='comparison', out_type='cnt', 
                                                       include=False, operation='rowmin', 
                                                       source=[variable1, variable2], 
                                                       destination=to_variable, 
                                                       parameters=parameters) 
    
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame   
    
    
def create_row_max_variable(DataFrame, variable1, variable2, to_variable=None, return_variable=False, return_script=False):
    
    if to_variable==None:
        to_variable = '{}MAX{}'.format(variable1,variable2)
    
    try:
        DataFrame[to_variable] = DataFrame[[variable1,variable2]].max(axis=1)
    except:
        DataFrame[to_variable] = None
        print('Row max({}, {}) Error : {}'.format(variable1, variable2, traceback.format_exc()))  

    parameters = {  }
    script_dict = generate_create_variable_task_script(type='comparison', out_type='cnt', 
                                                       include=False, operation='rowmax', 
                                                       source=[variable1, variable2], 
                                                       destination=to_variable, 
                                                       parameters=parameters) 
    
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame  
    
###############################################################################
##[ CREATING FEATURES - STR COMPARISON ]#######################################      
############################################################################### 
def create_str_comparison_variable(DataFrame, variable1, variable2, to_variable=None, operation='levenshtein', parameters={}, return_variable=False, return_script=False):    
    if to_variable==None:
        to_variable = '{}SIM{}'.format(variable1,variable2)
        
    try:
        case_sensitive = parameters['case_sensitive']
    except:
        case_sensitive = True
        
    if operation=='levenshtein':
        try:
            normalize = parameters['normalize']
        except:
            normalize = False
        DataFrame[to_variable] = np.vectorize(damerau_levenshtein_distance)(DataFrame[variable1], DataFrame[variable2], case_sensitive, normalize)

    elif operation=='jaccard':
        try:
            method=parameters['method']
        except:
            method='substring'
        try:
            min_length=parameters['min_length']
        except:
            min_length=1
        try:
            max_length=parameters['max_length']
        except:    
            max_length=np.inf
            
        DataFrame[to_variable] = np.vectorize(jaccard_index)(DataFrame[variable1], DataFrame[variable2], method, case_sensitive, min_length, max_length)

    script_dict = generate_create_variable_task_script(type='comparison_str', out_type='cnt', 
                                                       include=False, operation=operation, 
                                                       source=[variable1, variable2], 
                                                       destination=to_variable, 
                                                       parameters=parameters)
    
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame      

###############################################################################
##[ CREATING FEATURES - BINARY VARIABLES]######################################      
############################################################################### 
        
def create_binary_variable(DataFrame, to_variable, condition_str, default=0, null=0, return_variable=False, return_script=False):
    
    if to_variable==None:
        to_variable = '{}'.format(condition_str)

    try:    
        DataFrame[to_variable] = DataFrame.eval(condition_str).astype('int8').fillna(null)
        DataFrame.loc[DataFrame[to_variable].isna(), to_variable] = default
    except:
        print('Error in creating the binary variable {}:\n{}\n'.format(condition_str, traceback.format_exc()))
        print('Check variable rule set !')

    parameters = { 
                    'condition_str':condition_str,
                    'default': default,
                    'null': null
    }
    script_dict = generate_create_variable_task_script(type='condition', out_type='bin', 
                                                       include=False, operation='condition', 
                                                       source=None, 
                                                       destination=to_variable, 
                                                       parameters=parameters) 
            
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame 
    
###############################################################################
##[ CREATING FEATURES - CATEGORY LABELS]#######################################      
############################################################################### 
  
def num_label_to_value(num_label):
    units = {'p':0.000000000001,
        'n':0.000000001,
        'u':0.000001,
        'm':0.001,
        'c':0.01,
        'd':0.1,
        '':1,
        'D':10,
        'H':100,
        'K':1000,
        'M':1000000,
        'G':1000000000,
        'T':1000000000000,
        'P':1000000000000000,
        'E':1000000000000000000,
        'INF':np.inf        
        }
    try:
        sign, inf, num, unit = re.findall('^([-]?)((\d+)([pnumcdDHKMGTPE]?)|INF)$', num_label.rstrip().lstrip())[0]
        if inf=='INF':
            value = int('{}1'.format(sign))*np.inf
        else:
            value = int('{}1'.format(sign))*float(num)*units[unit]
    except:
        print('vnum_label_value failed !\n{}'.format(traceback.format_exc()))
        value = None
    return value

def edge_labels_to_values(edge_labels, left_inclusive=False, right_inclusive=False):
    """
    Parameters
    ----------
    edge_labels : str []
        Edge labels with number unit as postfix
            'p':0.000000000001,
            'n':0.000000001,
            'u':0.000001,
            'm':0.001,
            'c':0.01,
            'd':0.1,
            '':1,
            'D':10,
            'H':100,
            'K':1000,
            'M':1000000,
            'G':1000000000,
            'T':1000000000000,
            'P':1000000000000000,
            'INF':np.inf        
    left_inclusive : bool, default False
        Include left edge
    right_inclusive : bool, default False
        Include right edge
    
    Returns
    -------
    edge_values : numeric []
    bin_labels : str []
    """ 
    edge_values = []
    bin_labels = []
    n_bins = len(edge_labels)-1
    i=0
    for i in range(n_bins):        
        l_bracket = '(' if (i==0 and edge_labels[i]=='-INF') or (not left_inclusive) else '['
        r_bracket = ')' if (i==n_bins-1 and edge_labels[i+1]=='INF') or (not right_inclusive) else ']'
        edge_values.append(num_label_to_value(edge_labels[i]))
        bin_labels.append('{}_{}{},{}{}'.format(i+1, l_bracket, edge_labels[i], edge_labels[i+1], r_bracket))
    edge_values.append(num_label_to_value(edge_labels[n_bins]))
    return edge_values,bin_labels

###############################################################################
##[ CREATING FEATURES - CATEGORY]##############################################      
###############################################################################     
    
def create_categorical_variable(DataFrame, variable, to_variable, labels_str, right_inclusive=True, default='OTHER', null='NA', return_variable=False, return_script=False):
    
    if to_variable==None:
        to_variable = '{}GRP'.format(variable)
        
    try:
        default_ = '0_{}'.format(default)
        null_ = '0_{}'.format(null)
    except:
        default_ = '0_Other'
        null_ = '0_NA'

    edge_values, bin_labels = edge_labels_to_values(labels_str, left_inclusive=not right_inclusive, right_inclusive=right_inclusive)
    
    try:    
        DataFrame[to_variable] = pd.cut(DataFrame[variable], bins=edge_values, labels=bin_labels, right=right_inclusive, include_lowest=True).astype('object')
    except:
        DataFrame[to_variable] = null_

    DataFrame.loc[DataFrame[variable].isna(), to_variable] = null_
    DataFrame.loc[DataFrame[to_variable].isna(), to_variable] = default_

    parameters = { 
                    'labels_str':labels_str,
                    'right_inclusive': right_inclusive,
                    'default': default,
                    'null': null
    }
    script_dict = generate_create_variable_task_script(type='category', out_type='cat', 
                                                       include=False, operation='bucket', 
                                                       source=variable, 
                                                       destination=to_variable, 
                                                       parameters=parameters) 
        
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame 

def merge_categories(DataFrame, variable, to_variable, values, group_value, return_variable=False, return_script=False):
    if to_variable==None:
        to_variable = variable
    
    try:    
        DataFrame[to_variable] = DataFrame[variable].replace(to_replace=values, value=group_value)
    except:
        print('ERROR in creating the categorical variable merge {}:\n{}\n'.format(variable, traceback.format_exc()))
        print('Check variable rule set !')
        
    parameters = { 
                    'group_value':group_value,
                    'values': values
    }
    script_dict = generate_create_variable_task_script(type='category_merge', out_type='cat', 
                                                       include=False, operation='catmerge', 
                                                       source=variable, 
                                                       destination=to_variable, 
                                                       parameters=parameters)     

    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame  
    
###############################################################################
##[ CREATING FEATURES - ENTITY (DICTIONARY) ]##################################      
###############################################################################
def create_entity_variable(DataFrame, variable, to_variable, dictionary, match_type=None, default='OTHER', null='NA', return_variable=False, return_script=False):
    if to_variable==None:
        to_variable = '{}GRP'.format(variable)
              
    if to_variable != variable:
        DataFrame[to_variable] = None
    
    for entity in reversed(dictionary): 
        try:
            case=entity['case']
        except:
            case=True
                
        if (match_type=='values') or ('values' in  entity.keys()):
            if case==True:
                DataFrame.loc[DataFrame[variable].isin(entity['values']), to_variable] = entity['entity']
            else:
                values = [x.lower() for x in entity['values']] 
                DataFrame.loc[DataFrame[variable].str.lower().isin(values), to_variable] = entity['entity']
        elif (match_type=='pattern') or ('pattern' in entity.keys()):
            DataFrame.loc[DataFrame[variable].fillna('').str.contains(pat=entity['pattern'], case=case), to_variable] = entity['entity']
        else:
            print('Entity {} not created !'.format(entity))
            
    DataFrame.loc[DataFrame[variable].isna(), to_variable] = null
    DataFrame.loc[DataFrame[to_variable].isna(), to_variable] = default

    parameters = { 
                    'match_type':match_type,
                    'dictionary': dictionary,
                    'default': default,
                    'null': null
    }
    script_dict = generate_create_variable_task_script(type='entity', out_type='cat', 
                                                       include=False, operation='dictionary', 
                                                       source=variable, 
                                                       destination=to_variable, 
                                                       parameters=parameters)  
    
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame 

def create_value_pair_variable(DataFrame, variable1, variable2, to_variable, dictionary, match_type=None, default='OTHER', null='NA', return_variable=False, return_script=False):
    if to_variable==None:
        to_variable = '{}GRP{}'.format(variable1, variable2)
              
    if to_variable != variable1 and to_variable != variable2 :
        DataFrame[to_variable] = None
    
    for entity in reversed(dictionary): 
        try:
            case=entity['case']
        except:
            case=True
        
        try:
            opperator = entity['opperator']
        except:
            opperator='AND'
            
        if (match_type=='values') or ('values' in  entity.keys()):
            if case==True:
                if opperator=='AND':
                    DataFrame.loc[(DataFrame[variable1]==entity['values'][0]) & (DataFrame[variable2]==entity['values'][1]), to_variable] = entity['entity']
                elif opperator=='OR':
                    DataFrame.loc[(DataFrame[variable1]==entity['values'][0]) | (DataFrame[variable2]==entity['values'][1]), to_variable] = entity['entity']
                elif opperator=='NOT':
                    DataFrame.loc[(DataFrame[variable1]==entity['values'][0]) & (DataFrame[variable2]!=entity['values'][1]), to_variable] = entity['entity']
                elif opperator=='^NOT':
                    DataFrame.loc[(DataFrame[variable1]!=entity['values'][0]) & (DataFrame[variable2]==entity['values'][1]), to_variable] = entity['entity']
            else:
                values = [x.lower() for x in entity['values']] 
                if opperator=='AND':
                    DataFrame.loc[(DataFrame[variable1].str.lower()==values[0]) & (DataFrame[variable2].str.lower()==values[1]), to_variable] = entity['entity']
                elif opperator=='OR':
                    DataFrame.loc[(DataFrame[variable1].str.lower()==values[0]) | (DataFrame[variable2].str.lower()==values[1]), to_variable] = entity['entity']
                elif opperator=='NOT':
                    DataFrame.loc[(DataFrame[variable1].str.lower()==values[0]) & (DataFrame[variable2].str.lower()!=values[1]), to_variable] = entity['entity']
                elif opperator=='^NOT':
                    DataFrame.loc[(DataFrame[variable1].str.lower()!=values[0]) & (DataFrame[variable2].str.lower()==values[1]), to_variable] = entity['entity']
                                  
        elif (match_type=='pattern') or ('pattern' in entity.keys()):
            if opperator=='AND':
                DataFrame.loc[(DataFrame[variable1].fillna('').str.contains(pat=entity['values'][0], case=case)) & (DataFrame[variable2].fillna('').str.contains(pat=entity['values'][1], case=case)), to_variable] = entity['entity']
            elif opperator=='OR':
                DataFrame.loc[(DataFrame[variable1].fillna('').str.contains(pat=entity['values'][0], case=case)) | (DataFrame[variable2].fillna('').str.contains(pat=entity['values'][1], case=case)), to_variable] = entity['entity']                                  
        else:
            print('Entity {} not created !'.format(entity))
            
    DataFrame.loc[(DataFrame[variable1].isna()) & (DataFrame[variable2].isna()), to_variable] = null
    DataFrame.loc[DataFrame[to_variable].isna(), to_variable] = default

    parameters = { 
                    'match_type':match_type,
                    'dictionary': dictionary,
                    'default': default,
                    'null': null
    }
    script_dict = generate_create_variable_task_script(type='entity', out_type='cat', 
                                                       include=False, operation='valuepairs', 
                                                       source=[variable1, variable2],
                                                       destination=to_variable, 
                                                       parameters=parameters)  
    
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame 
    
###############################################################################
##[ CREATING FEATURES - PAIR EQUALITY ]########################################      
###############################################################################
def create_pair_equality_variable(DataFrame, variable1, variable2, to_variable, magnitude=False, case=True, return_variable=False, return_script=False):
    if to_variable==None:
        to_variable = '{}CMP{}'.format(variable1,variable2)
        
    DataFrame.loc[(DataFrame[variable1]==DataFrame[variable2]), to_variable] = 'EQ'
    DataFrame.loc[(DataFrame[variable1]!=DataFrame[variable2]), to_variable] = 'DF'
    DataFrame.loc[(DataFrame[variable1].isna()) | (DataFrame[variable2].isna()), to_variable] = 'ON'
    DataFrame.loc[(DataFrame[variable1].isna()) & (DataFrame[variable2].isna()), to_variable] = 'BN'

    parameters = { 
                    'magnitude':magnitude,
                    'case': case
    }
    script_dict = generate_create_variable_task_script(type='pair_equality', out_type='cat', 
                                                       include=False, operation='pairequality', 
                                                       source=[variable1, variable2], 
                                                       destination=to_variable, 
                                                       parameters=parameters)  
    
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame 

###############################################################################
        
###############################################################################
##[ CREATING FEATURES TASK - TARGET ]##########################################      
###############################################################################
def create_target_variable_task(DataFrame, rule_set, return_variable=False, return_script=False):
    to_variable = rule_set['variables']['destination']
    operation = rule_set['operation']    
    parameters = rule_set['parameters']
    
    target_condition_str = parameters['condition_str']
    default = parameters['default']
    null = parameters['null']
    
    DataFrame, to_variable, script_dict = set_binary_target(DataFrame, condition_str=target_condition_str, 
                                               to_variable=to_variable, default=default, null=null, return_variable=True, return_script=True)

    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame 
        
###############################################################################
##[ CREATING FEATURES TASK - TRANSFORM ]#######################################      
###############################################################################
def create_transformed_variable_task(DataFrame, rule_set, return_variable=False, return_script=False):
    variable = rule_set['variables']['source']
    to_variable = rule_set['variables']['destination']
    operation = rule_set['operation']    
    parameters = rule_set['parameters']
    
    if operation=='normalize':
        method = rule_set['parameters']['method']        
        DataFrame, to_variable, script_dict = create_normalized_variable(DataFrame, variable, method=method, parameters=parameters, to_variable=to_variable, return_variable=True, return_script=True)
    elif operation=='datepart':
        part = rule_set['parameters']['part']  
        DataFrame, to_variable, script_dict = create_datepart_variable(DataFrame, variable, part=part, to_variable=to_variable, return_variable=True, return_script=True)
    elif operation=='dateadd':
        unit = rule_set['parameters']['unit']  
        value = rule_set['parameters']['value']  
        DataFrame, to_variable, script_dict = create_dateadd_variable(DataFrame, variable, unit=unit, value=value, to_variable=to_variable, return_variable=True, return_script=True)
    elif operation=='log':
        base = rule_set['parameters']['base']  
        DataFrame, to_variable, script_dict = create_log_variable(DataFrame, variable, base=base, to_variable=to_variable, return_variable=True, return_script=True)
    elif operation=='exponent':
        base = rule_set['parameters']['base']  
        DataFrame, to_variable, script_dict = create_exponent_variable(DataFrame, variable, base=base, to_variable=to_variable, return_variable=True, return_script=True)
    elif operation=='exponent':
        a = rule_set['parameters']['a']  
        b = rule_set['parameters']['b'] 
        DataFrame, to_variable, script_dict = create_segmented_variable(DataFrame, variable, a=a, b=b, to_variable=to_variable, return_variable=True, return_script=True)
    else:
        pass # other transformations to be implemented
        
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame  


def create_str_transformed_variable_task(DataFrame, rule_set, return_variable=False, return_script=False):
    variable = rule_set['variables']['source']
    to_variable = rule_set['variables']['destination']
    operation = rule_set['operation']    
    parameters = rule_set['parameters']

    if operation=='strcount':
        pattern = parameters['pattern']
        case_sensitive = parameters['case_sensitive']
        DataFrame, to_variable, script_dict = create_str_count_variable(DataFrame, variable, pattern=pattern, case_sensitive=case_sensitive, to_variable=to_variable, return_variable=True, return_script=True)        
    elif operation=='normalize':
        to_case = parameters['to_case']
        chars = parameters['chars']
        numbers = parameters['numbers'] 
        spchar = parameters['spchar']
        space = parameters['space']        

        DataFrame, to_variable, script_dict = create_str_normalized_variable(DataFrame, variable, 
                                                                to_case=to_case, 
                                                                chars=chars, 
                                                                numbers=numbers, 
                                                                spchar=spchar, 
                                                                space=space, 
                                                                to_variable=None, return_variable=False, return_script=True)
    elif operation=='extract':
        pattern = parameters['pattern']
        case_sensitive = parameters['case_sensitive']
        DataFrame, to_variable, script_dict = create_str_extract_variable(DataFrame, variable, pattern=pattern, 
                                                             case_sensitive=case_sensitive, 
                                                             to_variable=to_variable, return_variable=True, return_script=True)   
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame      

###############################################################################
##[ CREATING FEATURES TASK - MLTI VARIAVLE ]###################################      
###############################################################################
        
def create_operation_mult_variable_task(DataFrame, rule_set, return_variable=False, return_script=False):
    variable = rule_set['variables']['source']
    to_variable = rule_set['variables']['destination']
    operation = rule_set['operation']    
    parameters = rule_set['parameters']    
    expression_str = parameters['expression_str']
    
    DataFrame, to_variable, script_dict = create_operation_mult_variable(DataFrame, expression_str=expression_str, 
                                                            to_variable=to_variable, return_variable=True, return_script=True)
    
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame 

###############################################################################
##[ CREATING FEATURES TASK - SEQUENCE ORDER ]##################################      
###############################################################################
def create_sequence_order_variable_task(DataFrame, rule_set, return_variable=False, return_script=False):
    variable1a = rule_set['variables']['source1a']
    variable2a = rule_set['variables']['source2a']
    variable1b = rule_set['variables']['source1b']
    variable2b = rule_set['variables']['source2b']
    to_variable = rule_set['variables']['destination']
    
    DataFrame, to_variable, script_dict = create_sequence_order_variable(DataFrame, variable1a, variable2a, variable1b, variable2b, output='binary', 
                                                            to_variable=to_variable, return_variable=True, return_script=True)
        
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame       
    
###############################################################################
##[ CREATING FEATURES TASK - COMPARISON ]######################################      
###############################################################################
def create_comparison_variable_task(DataFrame, rule_set, return_variable=False, return_script=False):
    variable1 = rule_set['variables']['source1']
    variable2 = rule_set['variables']['source2']
    to_variable = rule_set['variables']['destination']
    operation = rule_set['operation']
    parameters = rule_set['parameters']
    
    try:
        multiplier = parameters['multiplier']
    except:
        multiplier=1        
        
    try:
        unit = parameters['unit']
    except:
        unit = 'D'
    onerror = None # parameters['onerror']
    
    if operation=='numdiff':        
        DataFrame, to_variable, script_dict = create_numeric_difference_variable(DataFrame, variable1, variable2, multiplier=multiplier, onerror=onerror, to_variable=to_variable, return_variable=True, return_script=True)
    elif operation=='datediff':
        DataFrame, to_variable, script_dict = create_date_difference_variable(DataFrame, variable1, variable2, unit=unit, onerror=onerror, to_variable=to_variable, return_variable=True, return_script=True)
    elif operation=='rowmin':
        DataFrame, to_variable, script_dict = create_row_min_variable(DataFrame, variable1, variable2, to_variable=to_variable, return_variable=True, return_script=True)
    elif operation=='rowmax':
        DataFrame, to_variable, script_dict = create_row_max_variable(DataFrame, variable1, variable2, to_variable=to_variable, return_variable=True, return_script=True)    
    
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame       
    
def create_str_comparison_variable_task(DataFrame, rule_set, return_variable=False, return_script=False):
    variable1 = rule_set['variables']['source1']
    variable2 = rule_set['variables']['source2']
    to_variable = rule_set['variables']['destination']
    operation = rule_set['operation']    
    parameters = rule_set['parameters']
                
    DataFrame, to_variable, script_dict = create_str_comparison_variable(DataFrame, variable1=variable1, variable2=variable2, to_variable=to_variable, operation=operation, parameters=parameters, 
                                                                         return_variable=True, return_script=True)
    
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame 

###############################################################################
##[ CREATING FEATURES TASK - BINARY VARIABLE ]#################################      
###############################################################################
def create_binary_variable_task(DataFrame, rule_set, return_variable=False, return_script=False):
    #variable = rule_set['variables']['source']
    to_variable = rule_set['variables']['destination']
    parameters = rule_set['parameters']
    condition_str = parameters['condition_str']
    default = parameters['default']
    null = parameters['null']
    
    DataFrame, to_variable, script_dict = create_binary_variable(DataFrame, to_variable, condition_str, default, null, 
                                                                 return_variable=True, return_script=True)
    
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame 

###############################################################################
##[ CREATING FEATURES TASK - CATEGORY VARIABLE ]###############################      
###############################################################################  
def create_categorical_variable_task(DataFrame, rule_set, return_variable=False, return_script=False):  
    variable = rule_set['variables']['source']
    to_variable = rule_set['variables']['destination']
    operation = rule_set['operation']    
    parameters = rule_set['parameters']
    labels_str = parameters['labels_str']
    right_inclusive = parameters['right_inclusive'] 
    default = parameters['default']
    null = parameters['null']
    
    DataFrame, to_variable, script_dict = create_categorical_variable(DataFrame, variable, to_variable, labels_str, right_inclusive, default, null, 
                                                                      return_variable=True, return_script=True)
    
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame 

###############################################################################
##[ CREATING FEATURES TASK - ENTITY VARIABLE ]#################################      
###############################################################################         
def create_entity_variable_task(DataFrame, rule_set, return_variable=False, return_script=False):
    
    to_variable = rule_set['variables']['destination']
    parameters = rule_set['parameters']
    match_type = parameters['match_type']
    dictionary = parameters['dictionary'] 
    default = parameters['default']
    null = parameters['null']
    operation = rule_set['operation']

    if operation == 'dictionary':
        variable = rule_set['variables']['source']                                                               
        DataFrame, to_variable, script_dict = create_entity_variable(DataFrame, variable=variable, to_variable=to_variable, 
                                                                     dictionary=dictionary, match_type=match_type, default=default, null=null, 
                                                                     return_variable=True, return_script=True)
    elif operation == 'valuepairs':
        variable1 = rule_set['variables']['source1'] 
        variable2 = rule_set['variables']['source2'] 
        DataFrame, to_variable, script_dict = create_value_pair_variable(DataFrame, variable1, variable2, to_variable, 
                                                                     dictionary, match_type=None, default='OTHER', null='NA', 
                                                                     return_variable=True, return_script=True)

    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame       

###############################################################################
##[ CREATING FEATURES TASK - PAIR EQUALITY ]###################################      
############################################################################### 
def create_pair_equality_variable_task(DataFrame, rule_set, return_variable=False, return_script=False): 
    variable1 = rule_set['variables']['source1']
    variable2 = rule_set['variables']['source2']
    to_variable = rule_set['variables']['destination']
    parameters = rule_set['parameters']
    try:
        magnitude = parameters['magnitude']
    except:
        magnitude = 1
    case = parameters['case']
    
    DataFrame, to_variable, script_dict = create_pair_equality_variable(DataFrame, variable1=variable1, variable2=variable2, to_variable=to_variable, magnitude=magnitude, case=case, 
                                                                        return_variable=True, return_script=True)
    
    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame   

###############################################################################
##[ CREATING FEATURES TASK - MERGE CATEGORY ]##################################      
###############################################################################
def merge_categories_task(DataFrame, rule_set, return_variable=False, return_script=False):
    variable = rule_set['variables']['source']
    to_variable = rule_set['variables']['destination']
    values = rule_set['parameters']['values']
    group_value = rule_set['parameters']['group_value']
    
    DataFrame, to_variable, script_dict =  merge_categories(DataFrame, variable=variable, to_variable=to_variable, values=values, group_value=group_value, 
                                                            return_variable=True, return_script=True)   

    if return_script and return_variable:
        return DataFrame, to_variable, script_dict
    elif return_script:
        return DataFrame, script_dict
    elif return_variable:
        return DataFrame, to_variable
    else:
        return DataFrame 

###############################################################################
        
###############################################################################
##[ ENCODER ]##################################################################      
############################################################################### 
def to_one_hot_encode(DataFrame, category_variables=[], binary_variables=[], target_variable='target', target_type='binary'):
    # TO DO: If target type is 'multi' apply one hot encoding to target
    feature_variables = []
    try:
        VariablesDummies = pd.get_dummies(DataFrame[category_variables]).astype('int8')
        dummy_variables = list(VariablesDummies.columns.values)
        DataFrame[dummy_variables] = VariablesDummies
    except:
        print('Category columns {} does not specified nor exists'.format(category_variables))
        
    try:
        DataFrame[binary_variables] = DataFrame[binary_variables].astype('int8')
    except:
        print('Binary columns {} does not specified nor exists'.format(binary_variables))
    
    try:          
        feature_variables = binary_variables+dummy_variables
    except:
        print('Error in creating feature variables.')
        
    return DataFrame, feature_variables, target_variable

###############################################################################
##[ ML MODEL DRIVER ]##########################################################      
###############################################################################     
def load_data_task(load_data_dict, return_name=False):
    """
    Parameters
    ----------
    load_data_dict: dict
    e.g.:   {
        	  "type": "csv",
        	  "location": "local",
        	  "workclass": "Private",
        	  "source": {"path":"C:/Projects/Data/incomedata.csv", "separator":",", "encoding":null},
        	  "auth": None,
        	  "query": None,
        	  "limit": None
            }
    
    Returns
    -------
    DataFrame: pandas.DataFrame
    data_name: str
    """    

    import json
    if type(load_data_dict)==dict:
        pass
    else:
        try:
            load_data_dict = json.loads(load_data_dict) 
        except:
            print('ERROR in loading data:{}\n {}'.format(load_data_dict, traceback.format_exc()))  
    
    data_name = load_data_dict['data_name']
        
    if load_data_dict['type']=='csv':
        DataFrame = read_data_csv(
                file=load_data_dict['source']['path'], 
                separator=load_data_dict['source']['separator'], 
                encoding=load_data_dict['source']['encoding']
                )
    elif load_data_dict['type']=='pickle':
        DataFrame = read_data_pickle(
                file=load_data_dict['source']['path'], 
                compression =load_data_dict['source']['compression']
                )    
    elif load_data_dict['type']=='sql':
        DataFrame = read_data_sql(
                query=load_data_dict['query'], 
                server=load_data_dict['source']['server'], 
                database=load_data_dict['source']['database'],
                auth=load_data_dict['auth']
                )    
    else:
        print("No valid data source provided!")
        DataFrame = pd.DataFrame()	

    # Add ID column
    DataFrame = add_identity_column(DataFrame, id_label='ID', start=1, increment=1)

    # Clean column names
    DataFrame = clean_column_names(DataFrame, replace='')
        
    if return_name:  
        return DataFrame, data_name
    else:
        return DataFrame

###############################################################################
def create_variable_task(DataFrame, create_variable_task_dict=None, return_extra=False, return_script=False):
    """
    Interface function for single variable operation

    Parameters
    ----------
    DataFrame: pandas.DataFrame
    create_variable_task_dict : dict or JSON
    return_extra : bool, default False
        Returns variable_class and include if True
    
    Returns
    -------
    DataFrame: pandas.DataFrame
    data_name: str
    variable_class : str, optional
    include: bool, optional
    """    
    import json
    if type(create_variable_task_dict)==dict:
        pass
    else:
        try:
            create_variable_task_dict = json.loads(create_variable_task_dict) 
        except:
            print('ERROR in creating variable:{}\n {}'.format(create_variable_task_dict, traceback.format_exc()))  
            
    rule_set = {
        'operation':create_variable_task_dict['operation'],
        'variables':create_variable_task_dict['variables'],
        'parameters':create_variable_task_dict['parameters']
    }
    out_type = create_variable_task_dict['out_type']
    include = create_variable_task_dict['include']    

    try:
        if create_variable_task_dict['type']=='target':
            DataFrame, output_variable, script_dict  = create_target_variable_task(DataFrame, rule_set, return_variable=True, return_script=True)      
        if create_variable_task_dict['type']=='transform':
            DataFrame, output_variable, script_dict  = create_transformed_variable_task(DataFrame, rule_set, return_variable=True, return_script=True)  
        elif create_variable_task_dict['type']=='str_transform':
            DataFrame, output_variable, script_dict  = create_str_transformed_variable_task(DataFrame, rule_set, return_variable=True, return_script=True)          
        elif create_variable_task_dict['type']=='operation_mult':
            DataFrame, output_variable, script_dict  = create_operation_mult_variable_task(DataFrame, rule_set, return_variable=True, return_script=True)           
        elif create_variable_task_dict['type']=='seq_order':
            DataFrame, output_variable, script_dict  = create_sequence_order_variable_task(DataFrame, rule_set, return_variable=True, return_script=True)
        elif create_variable_task_dict['type']=='comparison':
            DataFrame, output_variable, script_dict  = create_comparison_variable_task(DataFrame, rule_set, return_variable=True, return_script=True)     
        elif create_variable_task_dict['type']=='str_comparison':
            DataFrame, output_variable, script_dict  = create_str_comparison_variable_task(DataFrame, rule_set, return_variable=True, return_script=True)    
        elif create_variable_task_dict['type']=='condition':
            DataFrame, output_variable, script_dict  = create_binary_variable_task(DataFrame, rule_set, return_variable=True, return_script=True)     
        elif create_variable_task_dict['type']=='category':
            DataFrame, output_variable, script_dict  = create_categorical_variable_task(DataFrame, rule_set, return_variable=True, return_script=True)    
        elif create_variable_task_dict['type']=='entity':
            DataFrame, output_variable, script_dict  = create_entity_variable_task(DataFrame, rule_set, return_variable=True, return_script=True)    
        elif create_variable_task_dict['type']=='pair_equality':
            DataFrame, output_variable, script_dict  = create_pair_equality_variable_task(DataFrame, rule_set, return_variable=True, return_script=True)   
        elif create_variable_task_dict['type']=='category_merge':
            DataFrame, output_variable, script_dict  = merge_categories_task(DataFrame, rule_set, return_variable=True, return_script=True)   
        else:
            output_variable= None    
            out_type = None
            include = False
            script_dict= {
                    "type": "",
                    "out_type":"",
                    "include": False,
                    "operation": "",
                    "variables": {
                        "source": "",
                        "destination": None
                    },
                    "parameters": {                
                    }
            }
    except:
        output_variable= None    
        out_type = None
        include = False
        script_dict= {
                "type": "",
                "out_type":"",
                "include": False,
                "operation": "",
                "variables": {
                    "source": "",
                    "destination": None
                },
                "parameters": {                
                }
        }        
    
    if return_script and return_extra:
        return DataFrame, output_variable, out_type, include, script_dict
    if return_script:
        return DataFrame, script_dict
    if return_extra:    
        return DataFrame, output_variable, out_type, include   
    else:
        return DataFrame, output_variable

def setup_variables_task(DataFrame, variables_setup_dict, return_script=False):
    """
    Parameters
    ----------
    DataFrame: pandas.DataFrame
    variables_setup_dict: json or dict
   
    
    Returns
    -------
    DataFrame: pandas.DataFrame
    category_variables: list(str)
    binary_variables: list(str)
    target_variable: list(str)
    """
    
    import re
    import json
    if type(variables_setup_dict)==dict:
        pass
    else:
        try:
            variables_setup_dict = json.loads(variables_setup_dict) 
        except:
            print('ERROR in creating variables:{}\n {}'.format(variables_setup_dict, traceback.format_exc()))  
            
    # Setting = {'model', 'score'}     
    setting = variables_setup_dict['setting']
    
    # verify if variables exists
    category_variables = variables_setup_dict['variables']['category_variables']
    binary_variables = variables_setup_dict['variables']['binary_variables']  
    target_variable = variables_setup_dict['variables']['target_variable'] 
    
    #Create variables sets
    category_variables =  set(category_variables) & set(DataFrame.columns)
    binary_variables  = set(binary_variables) & set(DataFrame.columns)
    
    # Create placeholder for variable creation scripts
    script_dict = []
    
    # Check if target variable exists (fill the column with None in scoring)
    if not target_variable in DataFrame.columns:
        DataFrame[target_variable]=None    
    
    # Run variable creation task list
    for preprocess_task in variables_setup_dict['preprocess_tasks']:
        task_type = preprocess_task['type'] #re.sub('[\W\d]', '', task_type)         
        if task_type in ['target', 'transform', 'condition', 'category', 'entity', 'category_merge', 'pair_equality', 'str_transform', 
                 'str_comparison', 'operation_mult', 'comparison', 'seq_order']:
            #print(task_type)
            
            DataFrame, variable_, variable_class_, include_, script_dict_ = create_variable_task(DataFrame, create_variable_task_dict=preprocess_task, return_extra=True, return_script=True)                    
   
            if include_:
                script_dict_['include'] = True
                script_dict.append(script_dict_)
                if variable_class_=='bin':
                    binary_variables.add(variable_)
                elif variable_class_=='cat':
                    category_variables.add(variable_)

    #Finalize variables lists
    category_variables=list(category_variables)
    binary_variables=list(binary_variables)
    target_variable = target_variable
    
    if return_script:
        return DataFrame, category_variables, binary_variables, target_variable, script_dict
    else:
        return DataFrame, category_variables, binary_variables, target_variable


###############################################################################
# Generate Script
###############################################################################
def generate_variables_script(source, destination):    
    if type(source)==list:
        if len(source)==2:
            variables = {
                'source1': source[0],
                'source2': source[1],
                'destination': destination
            }            
        elif len(source)==4:
            variables = {
                'source1a': source[0],
                'source2a': source[1],
                'source1b': source[2],
                'source2b': source[3],
                'destination': destination
            }
    else:
        variables = {
            'source': source,
            'destination': destination
        }
    return variables
    
def generate_create_variable_task_script(type='', out_type='', include=False, operation='', source=None, destination=None, parameters={}):
    variable_task_script = {
        'type': type,
        'out_type':out_type,
        'include': include,
        'operation': operation,
        'variables': generate_variables_script(source, destination),
        'parameters': parameters
    }
    return variable_task_script

###############################################################################
# EZ User Functions
###############################################################################
def create_category_ez(DataFrame, variable, labels_str, default='OTHER', null='NA', to_variable=None, target_variable=None, show_plot=True):
    rule_set = {   
        'operation':'bucket',
        'variables': {
            'source':variable, 
            'destination':to_variable
        },
        'parameters': {
            'labels_str': labels_str,
            'right_inclusive':True,
            'default':default,
            'null':null
        }
    }
    DataFrame, category_variable = mltk.create_categorical_variable_task(DataFrame, rule_set, return_variable=True)
    print(variable_response(DataFrame=DataFrame, variable=category_variable, target_variable=target_variable, show_plot=show_plot))
    return DataFrame, category_variable

def create_binary_ez(DataFrame, condition_str, default=0, null=0, to_variable=None, target_variable=None, show_plot=True):
    rule_set = {
        'operation':'condition',  
        'variables': {
            'source': None, 
            'destination':to_variable
        },
        'parameters': {
            'condition_str':condition_str,
            'default':default,
            'null':null,
        }
    } 
    
    DataFrame, binary_variable = create_binary_variable_task(DataFrame, rule_set, return_variables=True)    
    print(variable_response(DataFrame=DataFrame, variable=binary_variable, target_variable=target_variable, show_plot=show_plot))
    return DataFrame, binary_variable  

def create_entity_ez(DataFrame, variable, dictionary, default='OTHER', null='NA', to_variable=None, target_variable=None, show_plot=True):
    rule_set = {
        'operation':'dictionary',  
        'variables': {
            'source': variable, 
            'destination':to_variable
        },
        'parameters': {
            'match_type': None,
            'dictionary':dictionary,
            'default':default,
            'null':null,
        }
    } 
    
    DataFrame, entity_variable = create_entity_variable_task(DataFrame, rule_set, return_variables=True)    
    print(variable_response(DataFrame=DataFrame, variable=entity_variable, target_variable=target_variable, show_plot=show_plot))
    return DataFrame, entity_variable  

def create_entity_ez(DataFrame, variable, dictionary, default='OTHER', null='NA', to_variable=None, target_variable=None, show_plot=True):
    rule_set = {
        'operation':'dictionary',  
        'variables': {
            'source': variable, 
            'destination':to_variable
        },
        'parameters': {
            'match_type': None,
            'dictionary':dictionary,
            'default':default,
            'null':null,
        }
    } 
    
    DataFrame, entity_variable = create_entity_variable_task(DataFrame, rule_set, return_variables=True)    
    print(variable_response(DataFrame=DataFrame, variable=entity_variable, target_variable=target_variable, show_plot=show_plot))
    return DataFrame, entity_variable  