# -*- coding: utf-8 -*-
"""
SnowPy - A Python library to upload and doownload data from database systems
===============================================================================
- SnowPy data tools (datatools)
'SnowPy' was intially realsed as data tools (etl) for the mltoolkit project (https://mltoolkit.github.io/MLToolKit).

Author
------
- Sumudu Tennakoon

License
-------
- Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

Created Date
----------
- Sat Sep 28 2019

"""

from timeit import default_timer as timer
import gc
import socket
import getpass
import traceback
import sys
import os
import shutil
import csv
import urllib
import subprocess
import uuid
import pandas as pd
import numpy as np
try:
    import snowflake.connector
    from snowflake.sqlalchemy import URL
except:
    print("Error loading 'snowflake' library. Snowflake suppport disabled")
try:
    import sqlalchemy
except:
    print("Error loading 'sqlalchemy' library. database suppport disabled")
	
	
def print_data_resource_execute_format(): # Generate given use case (e.g. For SQL sever) in future version
    data_connector = {
        'type' : 'mssql', #{'snowflake', 'mysql', 'csv', 'json', 'pickle', 'hdf'}
        'connect_parameters' : {
            'file_path':None,
            'account' : None,
            'server': None,
            'database' : None,
            'schema' : None,
            'auth' : None # Only for sources with access permission # auth = {'type':'user', 'user':'user', 'password':'password', 'role': 'reader'}
        },
        'computing_parameters' : {
            'warehouse' : None, # For data warehouse services like Snowflake
            'driver' : '' # For SQL server connections
        }
    }
    
    data_object = {
        'identifiers' :{
            'file_name' : None,
            'table' : None, # Sor SQL
            'key' : None, # for HDF. Can also used to adress sheets in Excel
            'dataset_label' : None
        },
        'structure_parameters' : {
            'columns' : None,
            'index' : False,
            'header': 'infer',
            'partition_columns': None, # For parquet  Ref: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_parquet.html
            'orient' : None, # For JSON {'records', 'index', 'split} # Ref: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_json.html
        },
        'format_parameters': {
            'dtypes' : None,
            'date_columns' : None,
            'separator' : ',',
            'line_break' : '\r\n',
            'quoting' : 'ALL',
            'encoding' : None,
            'mode' : 'a', # for HDF {‘r’, ‘a’, 'w'}
            'compression' : 'infer', # 'infer' for Pickle # 'zlib'for HDF # Pickle, https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_pickle.html#pandas.DataFrame.to_pickle
            'file_format' : 'fixed', # for HDF {'fixed', 'table'}. Currently support only 'fixed'
            }
    }

    execute_params = {
        'chunksize' : None,
        'on_error' : 'ignore',
        'return_time'  : False, # time to execute
        'return_rowcount' : False, # rows affected
        'backend' : 'pandas' , #  to use pandas read/write methods
        'temp_file_path': '', # intermediate file for processing large dataset
        'read_params' : {
            'start' : None,
            'stop' : None
        },
        'write_params': {
            'if_exists' : 'fail',
            'insertion_method' : None, # If pandas {'multi'}
            'checksum' : False, # for HDF   
            'complevel' : 0, # for HDF                     
            'protocol' : -1  # Pickle https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_pickle.html#pandas.DataFrame.to_pickle
        }
    }

    print('data_connector =', data_connector)
    print('data_object =', data_object)
    print('execute_params =', execute_params)


def print_execute_time(execute_time, task_name=''): #convert ot use as decorator in future version
    try:
        execute_time = float(execute_time or 0)
    except:
        execute_time = 0.0
    print('{} time is {:.3f} s'.format(task_name, execute_time))

def print_records_count(records_count, task_name=''): #convert ot use as decorator in future version
    try:
        records_count = int(records_count or 0)
    except:
        records_count = 0
    print('{} {:,d} records'.format(task_name, records_count))

###############################################################################	
def set_field_value(value, default=None):
    """
    Parameters
    ----------
    value : str, int, float, object
    default :str, int, float, object, defualt None

    Returns
    -------
    value :str, int, float, object
    
    """
    try:
        return value
    except:
        return default

def get_field_value(fields_dict, key1=None, key2=None, default=None):
    """
    Parameters
    ----------
    fields_dict : dict
    key1 :str
    key2 : str
    default :str, int, float, object, defualt None
    
    Returns
    -------
    value :str, int, float, object
    """
    try:
        if key1!=None and key2!=None:
            value = set_field_value(fields_dict[key1][key2], default=default)
        elif key1!=None and key2==None:
            value = set_field_value(fields_dict[key1], default=default)
        else:
            raise Exception('Key error !')
        return value
    except:
        return default     
        
def read_data(query=None, data_connector=None, data_object =None, execute_params=None):
    """
    Parameters
    ----------
    query : str
    data_connector : dict
        run mltk.print_data_resource_execute_format() to get the format
    data_object : dict
        run mltk.print_data_resource_execute_format() to get the format
    execute_params : dict
        run mltk.print_data_resource_execute_format() to get the format
    Returns
    -------
    DataFrame : pandas.DataFrame
    """
    # Data Source Paramaters ###
    source_type = get_field_value(data_connector, 'type', default=None)
    server = get_field_value(data_connector, 'connect_parameters','server', default=None)
    database = get_field_value(data_connector, 'connect_parameters','server', default=None)
    auth = get_field_value(data_connector, 'connect_parameters','server', default=None)
    schema = get_field_value(data_connector, 'connect_parameters','server', default=None)
    warehouse = get_field_value(data_connector, 'computing_parameters','server', default=None)
    driver = get_field_value(data_connector, 'computing_parameters','driver', default='SQL Server')	
    #
    file_path = get_field_value(data_connector, 'connect_parameters','file_path', default='tempfile.tmp')
    # Data Object Parameters ###
    file_name = get_field_value(data_object, 'identifiers','file_name', default='tempfile.tmp')
    table = get_field_value(data_object, 'identifiers','table', default=None)
    key = get_field_value(data_object, 'identifiers','key', default='frame1')
    dataset_label  = get_field_value(data_object, 'identifiers','dataset_label', default=None)
    #
    columns  = get_field_value(data_object, 'structure_parameters','columns', default=None)
    index  = get_field_value(data_object, 'structure_parameters','index', default=False)
    header = get_field_value(data_object, 'structure_parameters','header', default='infer')
    partition_columns  = get_field_value(data_object, 'structure_parameters','partition_columns', default=None)
    orient  = get_field_value(data_object, 'structure_parameters','orient', default= 'records')
    #
    dtypes = get_field_value(data_object,'format_parameters','dtypes', default=None)
    date_columns = get_field_value(data_object,'format_parameters','date_columns', default=None)	
    separator = get_field_value(data_object,'format_parameters','separator', default=',')	
    line_break = get_field_value(data_object, 'format_parameters','line_break', default='\r\n')	
    quoting = get_field_value(data_object,'format_parameters','quoting', default='ALL')	
    encoding = get_field_value(data_object,'format_parameters','encoding', default='utf-8')	
    mode = get_field_value(data_object,'format_parameters','mode', default='a')
    compression = get_field_value(data_object, 'format_parameters','compression', default='infer')
    file_format = get_field_value(data_object, 'format_parameters','file_format', default='fixed')  
    # Execute Parameters ###
    chunksize = get_field_value(execute_params, 'chunksize', default=None)
    on_error = get_field_value(execute_params, 'on_error', default='ignore')
    return_time = get_field_value(execute_params, 'return_time', default=False)
    return_rowcount = get_field_value(execute_params, 'return_rowcount', default=False)
    backend = get_field_value(execute_params, 'backend', default='pandas')
    temp_file_path = get_field_value(execute_params, 'temp_file_path', default='')
    params = get_field_value(execute_params, 'params', default=None)     
    # Read Parameters	
    start = get_field_value(execute_params, 'read_params', 'start', default=None)
    stop = get_field_value(execute_params, 'read_params', 'stop', default=None)
    # Write Parameters
    if_exists = get_field_value(execute_params, 'write_params', 'if_exists', default='fail')
    insertion_method = get_field_value(execute_params, 'write_params','insertion_method', default='multi')
    complevel = get_field_value(execute_params, 'write_params', 'complevel', default=0)
    checksum = get_field_value(execute_params, 'write_params', 'checksum', default=False)
    protocol = get_field_value(execute_params, 'write_params', 'protocol', default=0)
    partition_columns = get_field_value(execute_params, 'write_params', 'partition_columns', default=None)
    
    if source_type == 'mssql':
        read_data_mssql(
            query=query, 
            server=server, 
            database=database, 
            auth=auth, 
            driver=driver, 
            on_error=on_error, 
            return_time=return_time, 
            params=params
            )	
    elif source_type == 'csv':
        print(source_type, file_path)
        DataFrame = read_data_csv(
            file=file_path, 
            separator=separator, 
            quoting=quoting , 
            compression=compression, 
            encoding=encoding, 
            on_error=on_error, 
            return_time=return_time
            )
    elif source_type == 'pickle':
        DataFrame = read_data_pickle(
            file=file_path,
            compression=compression
            )
    elif source_type == 'hdf':
        DataFrame = read_data_hdf(
            file=file_path, 
            key=key, 
            mode=mode, 
            start=start, 
            stop=stop, 
            header=header,
            columns=columns, 
            on_error=on_error
            )
    elif source_type == 'snowflake':
        DataFrame = read_data_snowflake(
            query=query, 
            server=server, 
            database=database, 
            auth=auth, 
            schema=schema, 
            warehouse=warehouse, 
            date_columns=date_columns, 
            chunksize=chunksize, 
            backend=backend, 
            on_error=on_error, 
            return_time=return_time
        )
    elif source_type == 'parquet':
        DataFrame = read_data_parquet(
            file=file_path,
            engine=engine, 
            columns=columns, 
            on_error=on_error
            )
    else:
        print('source_type {} not supported'.format(source_type))
        DataFrame = pd.DataFrame()
    
    return DataFrame

def write_data(DataFrame=None, data_connector=None, data_object =None, execute_params=None):		
    """
    Parameters
    ----------
    DataFrame : pandas.DataFrame
    data_resource : dict
        run mltk.print_data_resource_execute_format() to get the format
    execute_params : dict
        run mltk.print_data_resource_execute_format() to get the format

    Returns
    -------
    None
    """ 
    # Data Source Paramaters ###
    source_type = get_field_value(data_connector, 'type', default=None)
    server = get_field_value(data_connector, 'connect_parameters','server', default=None)
    database = get_field_value(data_connector, 'connect_parameters','server', default=None)
    auth = get_field_value(data_connector, 'connect_parameters','server', default=None)
    schema = get_field_value(data_connector, 'connect_parameters','server', default=None)
    warehouse = get_field_value(data_connector, 'computing_parameters','server', default=None)
    driver = get_field_value(data_connector, 'computing_parameters','driver', default='SQL Server')	
    #
    file_path = get_field_value(data_connector, 'connect_parameters','file_path', default='tempfile.tmp')
    # Data Object Parameters ###
    file_name = get_field_value(data_object, 'identifiers','file_name', default='tempfile.tmp')
    table = get_field_value(data_object, 'identifiers','table', default=None)
    key = get_field_value(data_object, 'identifiers','key', default='frame1')
    dataset_label  = get_field_value(data_object, 'identifiers','dataset_label', default=None)
    #
    columns  = get_field_value(data_object, 'structure_parameters','columns', default=None)
    index  = get_field_value(data_object, 'structure_parameters','index', default=False)
    header = get_field_value(data_object, 'structure_parameters','header', default=True)
    partition_columns  = get_field_value(data_object, 'structure_parameters','partition_columns', default=None)
    orient  = get_field_value(data_object, 'structure_parameters','orient', default= 'records')
    #
    dtypes = get_field_value(data_object,'format_parameters','dtypes', default=None)
    date_columns = get_field_value(data_object,'format_parameters','date_columns', default=None)	
    separator = get_field_value(data_object,'format_parameters','separator', default=',')	
    line_break = get_field_value(data_object, 'format_parameters','line_break', default='\r\n')	
    quoting = get_field_value(data_object,'format_parameters','quoting', default='ALL')	
    encoding = get_field_value(data_object,'format_parameters','encoding', default='utf-8')	
    mode = get_field_value(data_object,'format_parameters','mode', default='a')
    compression = get_field_value(data_object, 'format_parameters','compression', default='infer')
    file_format = get_field_value(data_object, 'format_parameters','file_format', default='fixed')  
    # Execute Parameters ###
    chunksize = get_field_value(execute_params, 'chunksize', default=None)
    on_error = get_field_value(execute_params, 'on_error', default='ignore')
    return_time = get_field_value(execute_params, 'return_time', default=False)
    return_rowcount = get_field_value(execute_params, 'return_rowcount', default=False)
    backend = get_field_value(execute_params, 'backend', default='pandas')
    temp_file_path = get_field_value(execute_params, 'temp_file_path', default='')
    params = get_field_value(execute_params, 'params', default=None)     
    # Read Parameters	
    start = get_field_value(execute_params, 'read_params', 'start', default=None)
    stop = get_field_value(execute_params, 'read_params', 'stop', default=None)
    # Write Parameters
    if_exists = get_field_value(execute_params, 'write_params', 'if_exists', default='fail')
    insertion_method = get_field_value(execute_params, 'write_params','insertion_method', default='multi')
    complevel = get_field_value(execute_params, 'write_params', 'complevel', default=0)
    checksum = get_field_value(execute_params, 'write_params', 'checksum', default=False)
    protocol = get_field_value(execute_params, 'write_params', 'protocol', default=0)
    partition_columns = get_field_value(execute_params, 'write_params', 'partition_columns', default=None)
    
    if source_type == 'mssql':
        write_data_mssql(
            DataFrame, 
            server=server, 
            database=database, 
            schema=schema, 
            table=table, 
            index=index, 
            dtypes=dtypes, 
            if_exists=if_exists, 
            auth=auth, 
            insertion_method=insertion_method, 
            chunksize=chunksize,
            driver=driver, 
            on_error=on_error, 
            return_time=return_time, 
            params=params
            )
    elif source_type == 'csv':
        write_data_csv(
            DataFrame=DataFrame, 
            file=file_path, 
            separator=separator, 
            index=index, 
            header=header,
            columns=columns,
            quoting=quoting, 
            encoding=encoding, 
            compression=compression, 
            chunksize=chunksize, 
            on_error=on_error, 
            return_time=return_time
            )
    elif source_type == 'pickle':
        write_data_pickle(
            DataFrame=DataFrame, 
            file=file_path, 
            compression=compression, 
            protocol=protocol
            )
    elif source_type == 'hdf': 
        write_data_hdf(
            DataFrame=DataFrame,
            file=file_path, 
            key=key, 
            mode=mode, 
            file_format=file_format,
            columns=columns, 
            compression=compression,
            complevel=complevel,
            checksum=checksum,
            if_exists=if_exists, 
            on_error=on_error
            )
    elif source_type == 'snowflake':
        write_data_snowflake(
            DataFrame=DataFrame, 
            server=server, 
            database=database, 
            auth=auth, 
            schema=schema, 
            table=table, 
            columns=columns, 
            warehouse=warehouse, 
            index=index, 
            if_exists=if_exists, 
            chunksize=chunksize, 
            dtypes=dtypes, 
            insertion_method=insertion_method, 
            temp_file_path=temp_file_path, 
            dataset_label=dataset_label,  
            backend=backend, 
            on_error=on_error, 
            return_time=return_time
            )
    elif source_type == 'parquet':
        write_data_parquet(
            DataFrame=DataFrame, 
            file=file_path, 
            engine=engine, 
            compression=compression, 
            index=index, 
            partition_columns=partition_columns, 
            if_exists=if_exists, 
            on_error=on_error
            )
    else:
        print('source_type {} not supported'.format(source_type))
        
    return None

def read_data_parquet(file, engine='auto', columns=None, on_error='strict'):
    """
    Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_hdf.html
    
    Parameters
    ----------    
    file : str
    engine :  {‘auto’, ‘pyarrow’, ‘fastparquet’}, default 'auto'
    columns : list, default None
    on_error : {'strict', errors, }, default 'strict' Ref: https://docs.python.org/3/library/functions.html#open
    
    Returns
    -------
    DataFrame : pandas.DataFrame
    """
    try:
        start_time = timer() 
        DataFrame = pd.read_parquet(path=file, engine=engine, columns=columns)
        execute_time = timer() - start_time
        rowcount = len(DataFrame.index)
    except:
        execute_time = 0
        rowcount = 0
        print(traceback.format_exc())
        DataFrame = pd.DataFrame()
    
    print_execute_time(execute_time, task_name='read')
    print_records_count(records_count=rowcount, task_name='read') 
        
    return DataFrame

def write_data_parquet(DataFrame, file, engine='auto', compression='snappy', index=None, partition_columns=None, if_exists=None, on_error='strict'):
    """
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_hdf.html#pandas.DataFrame.to_hdf
            compression=compression,
            complevel=complevel,   
    Parameters
    ----------    
    DataFrame : pandas.DataFrame
    file : str
    engine :  {‘auto’, ‘pyarrow’, ‘fastparquet’}, default 'auto'
    compression : {‘snappy’, ‘gzip’, ‘brotli’, None}, default ‘snappy’
    index : boot, default None
    partition_columns : list, default None
    if_exists : str, default None
    on_error : {'strict', errors'}, default 'strict' Ref: https://docs.python.org/3/library/functions.html#open

    Returns
    -------
    None    
    """

    if compression == 'infer' or compression==None:
        compression = 'snappy'

    try:
        start_time = timer() 
        DataFrame.to_parquet(fname=file, engine=engine, compression=compression, index=index, partition_cols=partition_columns)
        execute_time = timer() - start_time
        rowcount = len(DataFrame.index)
    except:
        execute_time = 0
        rowcount = 0
        print(traceback.format_exc())
    
    print_execute_time(execute_time, task_name='write')
    print_records_count(records_count=rowcount, task_name='write') 
    
    
def read_data_hdf(file, key='frame', mode='a', start=None, stop=None, columns=None, on_error='strict'):
    """
    Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_hdf.html
    
    Parameters
    ----------    
    file : str
    key : object, default 'frame'
    mode : {'r', 'r+', 'a'} default 'a'
    start : int, default None
    stop : int, default None
    columns : list, default None
    on_error : {'strict', errors, }, default 'strict' Ref: https://docs.python.org/3/library/functions.html#open
    
    Returns
    -------
    DataFrame : pandas.DataFrame
    """
    try:
        start_time = timer() 
        DataFrame = pd.read_hdf(path_or_buf=file, key=key, mode=mode, start=start, stop=stop, columns=columns, errors=on_error)
        execute_time = timer() - start_time
        rowcount = len(DataFrame.index)
    except:
        execute_time = 0
        rowcount = 0
        print(traceback.format_exc())
        DataFrame = pd.DataFrame()
    
    print_execute_time(execute_time, task_name='read')
    print_records_count(records_count=rowcount, task_name='read') 
        
    return DataFrame

def write_data_hdf(DataFrame, file, key='frame', mode='a', file_format='fixed', columns=None, compression='zlib', complevel=0, checksum=False, if_exists=None, on_error='strict'):
    """
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_hdf.html#pandas.DataFrame.to_hdf
            compression=compression,
            complevel=complevel,   
    Parameters
    ----------    
    DataFrame : pandas.DataFrame
    file : str
    key : object, default 'frame'
    mode : {'w', 'r+', 'a'} default 'w'
    format {'fixed', 'table'}, default 'fixed'
    if_exists : str, default None
    columns : list, default None
    compression : {'zlib', 'lzo', 'bzip2', 'blosc'}, default 'zlib'
    complevel : int , default 0
    checksum : bool, default False
    on_error : {'strict', errors'}, default 'strict' Ref: https://docs.python.org/3/library/functions.html#open

    Returns
    -------
    None    
    """
    if if_exists == 'append':
        append  = True
    else:
        append  = False

    if compression == 'infer' or compression==None:
        compression = 'zlib'

    try:
        start_time = timer() 
        DataFrame.to_hdf(path_or_buf=file, key=key, mode=mode, format=file_format, if_exists=if_exists, data_columns=columns, checksum=checksum, complevel=complevel, complib=compression, on_error=on_error)
        execute_time = timer() - start_time
        rowcount = len(DataFrame.index)
    except:
        execute_time = 0
        rowcount = 0
        print(traceback.format_exc())
    
    print_execute_time(execute_time, task_name='write')
    print_records_count(records_count=rowcount, task_name='write') 
	
	
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
        rowcount = len(DataFrame.index)
    except:
        execute_time = 0
        rowcount = 0
        print(traceback.format_exc())
        DataFrame = pd.DataFrame()
    
    print_execute_time(execute_time, task_name='read')
    print_records_count(records_count=rowcount, task_name='read') 
    
    return DataFrame

def write_data_pickle(DataFrame, file, compression='infer', protocol=3, return_time=False):
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
    return_time : bool, default False
    
    Returns
    -------
    None    
    """
    try:
        start_time = timer() 
        DataFrame.to_pickle(path=file, compression=compression, protocol=protocol)
        execute_time = timer() - start_time
        rowcount = len(DataFrame.index)
    except:
        execute_time = 0
        rowcount = 0
        rowcount =None
        print(traceback.format_exc())
        
    print_execute_time(execute_time, task_name='write')
    print_records_count(records_count=rowcount, task_name='write') 

    if return_time:
        return rowcount, execute_time
    else:
        return rowcount 
        
def execute_snowflake_sql_query(query, server=None, database=None, auth=None, schema=None, 
                                warehouse=None, on_error='ignore', return_time=False):         
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
        e.g. auth = {'type':'user', 'user':'user', 'password':'password', 'role': 'reader'} for username password authentication
             auth = {'type':'machine', 'user':None, 'password':None} for machine authentication
    schema : str 
    warehouse : str
           
    Returns
    -------
    None
    """    
    
    engine = sqlalchemy.create_engine(URL(
        account = server,
        user = auth['user'],
        password = auth['password'],
        database = database,
        schema = schema,
        warehouse = warehouse,
        role = auth['role'],
    ))  
    
    try:   
        with engine.connect() as connection:
            start_time = timer() 
            connection.execute(query)
            execute_time = timer() - start_time
            print('Execute Time:', float(execute_time or 0)) 
            rowcount = None # need to add return value for affected rows
    except:
        execute_time = None
        rowcount = None
        
    if return_time:
        return rowcount, execute_time
    else:
        return rowcount  

                          
def read_data_snowflake(query, server=None, database=None, auth=None, schema=None, warehouse=None, 
                        date_columns=None, chunksize=None, backend='pandas', on_error='ignore', return_time=False):
    """
    Parameters
    ----------
    query : str
        SQL query
    server : str
    database : str 
    auth :  dict
        e.g. auth = {'type':'user', 'user':'user', 'password':'password'} for username password authentication
             auth = {'type':'machine', 'uid':None, 'pwd':None} for machine authentication
    schema : str
    warehouse : str
    date_columns : list(str)
    chunksize : int
    backend : str  
    
    Returns
    -------
    DataFrame : pandas.DataFrame
    """    
    
    execute_time = None
    
    engine = sqlalchemy.create_engine(URL(
        account = server,
        user = auth['user'],
        password = auth['password'],
        database = database,
        schema = schema,
        warehouse = warehouse,
        role = auth['role'],
    ))  
    
    try:
        with engine.connect() as connection:
            start_time = timer()
            if backend == 'pandas':
                DataFrame = pd.read_sql(query, con=connection, parse_dates=date_columns, chunksize=chunksize)
            else:
                print("No backend provided. Supported backends: 'pandas'")
            execute_time = timer() - start_time
    except:
        DataFrame = pd.DataFrame()

    rowcount = len(DataFrame.index)
    
    print_execute_time(execute_time, task_name='read')
    print_records_count(records_count=rowcount, task_name='read') 
            
    if return_time:
        return DataFrame, execute_time
    else:
        return DataFrame           
    
def write_data_snowflake(DataFrame, server=None, database=None, auth=None, schema=None, table=None, columns=None, 
                         warehouse=None, index=False, if_exists='fail', chunksize=None, dtypes=None, insertion_method='multi', 
                         temp_file_path='', dataset_label=None,  backend='pandas', on_error='ignore', return_time=False):
    """
    Parameters
    ----------
    DataFrame : pandas.DataFrame
    server : str
        Database Server
    database : str
        Database
    auth :  dict
        e.g. auth = {'type':'user', 'user':'user', 'password':'password'} for username password authentication
             auth = {'type':'machine', 'uid':None, 'pwd':None} for machine authentication
    schema : str    
    table : str
    columns : list(str)
    warehouse : str
    index : bool
    if_exists : {'fail', 'append', 'replace'}, default 'fail'
    chunksize : int
    dtypes : list(str)
    insertion_method : str, default 'multi'
    temp_file_path : str
    dataset_label : str
    backend : {'pandas|'bulk'} 
    on_error : {None, 'ignore', 'raise'}, default 'ignore'
    return_time : bool, default False
    
    Returns
    -------
    None
    """   
    
    if columns!=None:
        DataFrame = DataFrame[columns]
        
    column_names = DataFrame.columns    
    column_numbers = np.arange(len(column_names))+1
    
    COLUMN_NAME_LIST = ','.join(column_names.str.upper().values)
    COLUMN_NUMBERS_LIST = ','.join(['${}'.format(n) for n in column_numbers])
    
    engine = sqlalchemy.create_engine(URL(
        account = server,
        user = auth['user'],
        password = auth['password'],
        database = database,
        schema = schema,
        #warehouse = warehouse,
        role = auth['role'],
    ))  
    
    rowcount = None
    execute_time = None
            
    try:    
        with engine.connect() as connection:
            start_time = timer()        
            use_warehouse_command = """
            USE WAREHOUSE {warehouse};
            """.format(warehouse=warehouse)
            
            suspend_warehouse_command = """
            ALTER WAREHOUSE IF EXISTS {warehouse} SUSPEND;
            """.format(warehouse=warehouse)
            
            if backend == 'pandas':
                connection.execute(use_warehouse_command)
                DataFrame.to_sql(table, con=connection, schema=schema, if_exists=if_exists, index=index, 
                                 dtype=dtypes, method=insertion_method, chunksize=chunksize)
                connection.execute(suspend_warehouse_command)
            elif backend == 'bulk':
                DataFrame.to_csv(temp_file_path, index=False, header=False, quoting=csv.QUOTE_ALL)
                
                stage_name = 'snowpy_stage_2019'
                file_format_name = 'snowpy_file_format'
                
                file_format_command = """
                CREATE OR REPLACE FILE FORMAT {file_format_name}
                TYPE = CSV
                FIELD_OPTIONALLY_ENCLOSED_BY ='"'
                FIELD_DELIMITER = ','
                SKIP_HEADER = 0
                NULL_IF = ('');
                """.format(file_format_name=file_format_name)
                
                stage_create_command = """
                CREATE OR REPLACE STAGE {stage_name}
                FILE_FORMAT = {file_format_name};            
                """.format(file_format_name=file_format_name, stage_name=stage_name)
            
                stage_drop_command = """
                DROP STAGE {stage_name};
                """.format(stage_name=stage_name)
    
                file_upload_command = """
                PUT file://{temp_file_path} @{stage_name}
                PARALLEL = 32;
                """.format(temp_file_path=temp_file_path, stage_name=stage_name)
    
                insert_command = r"""
                COPY INTO {table} ({COLUMN_NAME_LIST})
                FROM (
                  SELECT {COLUMN_NUMBERS_LIST}
                  FROM @{stage_name}/{table_name}.csv.gz t
                )
                PURGE = TRUE;
                """.format(table=table, stage_name=stage_name,
                COLUMN_NAME_LIST=COLUMN_NAME_LIST,
                COLUMN_NUMBERS_LIST=COLUMN_NUMBERS_LIST
                )
                
                connection.execute(stage_create_command)
                connection.execute(file_format_command)
                connection.execute(file_upload_command)
                #
                connection.execute(use_warehouse_command)          
                connection.execute(insert_command)
                connection.execute(stage_drop_command)
                connection.execute(suspend_warehouse_command)
                
            rowcount = DataFrame.shape[0]            
            execute_time = timer() - start_time
            print('data write time:', float(execute_time or 0))  

    except:
        execute_time = None
        rowcount = None

    print_execute_time(execute_time, task_name='read')
    print_records_count(records_count=rowcount, task_name='read') 
    
    if return_time:
        return rowcount, execute_time
    else:
        return rowcount 
    
def execute_mssql_query(query=None, server=None, database=None, auth=None, driver='ODBC Driver 13 for SQL Server', 
                         on_error='ignore', return_time=False, params=None):
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
        e.g. auth = {'type':'user', 'user':'user', 'password':'password'} for username password authentication
             auth = {'type':'machine', 'uid':None, 'pwd':None} for machine authentication
    params : dict
        extra parameters (not implemented)
    driver : str, default 'ODBC Driver 13 for SQL Server'
        E.g.: 'ODBC Driver 13 for SQL Server', 'ODBC Driver 17 for SQL Server', 'SQL Server'
    on_error : {None, 'ignore', 'raise'}, default 'ignore'
    return_time : bool
        
    Returns
    -------
    DataFrame : pandas.DataFrame
    """        
        
    # Download ODBC Driver https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server
    # 'SQL Server' # 
    autocommit = 'True'
    fast_executemany = True
    
    if server!=None and  database!=None and query!=None and auth!=None :
        try:
            if auth['type']=='machine':
                connect_string = r'Driver={'+driver+'};SERVER='+server+';DATABASE='+database+';TRUSTED_CONNECTION=yes;autocommit='+autocommit+';'
                connect_string = urllib.parse.quote_plus(connect_string)
                
            elif auth['type']=='user':
                user =  auth['user'] 
                password =  auth['password'] 
                connect_string = r'Driver={'+driver+'};SERVER='+server+';DATABASE='+database+';UID='+user+'r;PWD='+password+'; autocommit='+autocommit+';'
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
                print('{} rows affected. execute time = {} s'.format(int(rowcount or 0), float(execute_time or 0)))
            except:
                rowcount = None
                print('ERROR in fetching affected rows count. execute time = {} s'.format(float(execute_time or 0)))
                
            # commit
            trans.commit()
        
            # close connections, results set and dispose engine (moved to finally)
            #connection.close()
            #result.close()
            #engine.dispose()
        except:
            print(r'ERROR: Check If ODBC driver installed. \nIf not, Download ODBC Driver from https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server:\n{}\n'.format(traceback.format_exc()))
            rowcount = None
            execute_time = None
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
    else:
        execute_time = None
        rowcount = None 
        
    if return_time:
        return rowcount, execute_time
    else:
        return rowcount     

    
def read_data_mssql(query=None, server=None, database=None, auth=None, driver='SQL Server',  
                    on_error='ignore', return_time=False, params=None):
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
        e.g. auth = {'type':'user', 'user':'user', 'password':'password'} for username password authentication
             auth = {'type':'machine', 'uid':None, 'pwd':None} for machine authentication
    driver : str, default 'ODBC Driver 13 for SQL Server'
        E.g.: 'ODBC Driver 13 for SQL Server', 'ODBC Driver 17 for SQL Server', 'SQL Server'
    on_error : {None, 'ignore', 'raise'}, default 'ignore'
    return_time : bool
    
    Returns
    -------
    DataFrame : pandas.DataFrame
    """   
    # Download ODBC Driver https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server
        
    autocommit = 'True'
    fast_executemany = True
    execute_time = None

    if server!=None and  database!=None and auth!=None : 
        coerce_float=True
        index_col=None
        parse_dates=None
        
        try:
            if auth['type']=='machine':
                #connect_string = r'Driver={SQL Server};SERVER='+server+';DATABASE='+database+';TRUSTED_CONNECTION=yes;' #ODBC (slow)
                connect_string = r'Driver={'+driver+'};SERVER='+server+';DATABASE='+database+';TRUSTED_CONNECTION=yes;autocommit='+autocommit+';'
                connect_string = urllib.parse.quote_plus(connect_string)
            elif auth['type']=='user':
                user =  auth['user'] 
                password =  auth['password'] 
                #connect_string = r'Driver={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+uid+'r;PWD='+pwd+'}' #ODBC (slow)
                connect_string = r'Driver={'+driver+'};SERVER='+server+';DATABASE='+database+';UID='+user+'r;PWD='+password+'; autocommit='+autocommit+';'
                connect_string = urllib.parse.quote_plus(connect_string)
            else:
                raise Exception('No db server authentication method provided !') 
                
            #connection = pyodbc.connect(connect_string) #ODBC (slow)
            engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect="+connect_string, fast_executemany=fast_executemany)
            connection = engine
            
            start_time = timer() 
            DataFrame = pd.read_sql_query(sql=query, con=connection, coerce_float=coerce_float, index_col=index_col, parse_dates=parse_dates)
            execute_time = timer() - start_time
            rowcount = len(DataFrame.index)
            
            #connection.close() 
            engine.dispose()
        except:
            print('Database Query Failed! Check If ODBC driver installed. \nIf not, Download ODBC Driver from https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-serve.:\n{}\n'.format(traceback.format_exc()))
            execute_time = None
            rowcount = None
    else:
        print('Check the destiniation table path (server, database, schema, table, auth) !')
        DataFrame=pd.DataFrame()
        execute_time = None
        rowcount = None
    
    print_execute_time(execute_time, task_name='read')
    print_records_count(records_count=rowcount, task_name='read') 
    
    if return_time:
        return DataFrame, execute_time
    else:
        return DataFrame


def write_data_mssql(DataFrame, server=None, database=None, schema=None, table=None, index=False, 
                     dtypes=None, if_exists='fail', auth=None, insertion_method=None, chunksize=None,
                     driver='ODBC Driver 13 for SQL Server', on_error='ignore', return_time=False, params=None):
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
        e.g. auth = {'type':'user', 'user':'user', 'password':'password'} for username password authentication
             auth = {'type':'machine', 'uid':None, 'pwd':None} for machine authentication
    insertion_method : {None, 'multi', callable}
        see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_sql.html
    driver : str, default 'ODBC Driver 13 for SQL Server'
        E.g.: 'ODBC Driver 13 for SQL Server', 'ODBC Driver 17 for SQL Server', 'SQL Server'
    on_error : {None, 'ignore', 'raise'}, default 'ignore'
    return_time : bool
    
    Returns
    -------
    None
    """ 
    
    # Download ODBC Driver https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server
        
    autocommit = 'True'
    fast_executemany = True
    
    if server!=None and  database!=None and schema!=None and table!=None and auth!=None : 
        try:
            if auth['type']=='machine':
                connect_string = r'Driver={'+driver+'};SERVER='+server+';DATABASE='+database+';TRUSTED_CONNECTION=yes;autocommit='+autocommit+';'
                connect_string = urllib.parse.quote_plus(connect_string)
            elif auth['type']=='user':
                user =  auth['user'] 
                password =  auth['password'] 
                connect_string = r'Driver={'+driver+'};SERVER='+server+';DATABASE='+database+';UID='+user+'r;PWD='+password+'; autocommit='+autocommit+';'
                connect_string = urllib.parse.quote_plus(connect_string)
            else:
                raise Exception('No db server authentication method provided !') 
                
            #connection = pyodbc.connect(connect_string) #ODBC (slow)
            engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect="+connect_string, fast_executemany=fast_executemany)
            connection = engine
            
            start_time = timer() 
            if dtypes==None:
                DataFrame.to_sql(name=table, con=connection, schema=schema, index= index, if_exists=if_exists, method=insertion_method)
            else:
                DataFrame.to_sql(name=table, con=connection, schema=schema, index= index, dtype=dtypes, if_exists=if_exists, method=insertion_method)
            execute_time = timer() - start_time
            
            #connection.close() 
            engine.dispose()
            rowcount = len(DataFrame.index)
        except:
            print('Database Query Failed! Check If ODBC driver installed. \nIf not, Download ODBC Driver from https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-serve.:\n{}\n'.format(traceback.format_exc()))
            rowcount = None
            execute_time = None
    else:
        print('Check the destiniation table path (server, database, schema, table, auth) !')
        rowcount = None
        execute_time = None
    
    print_execute_time(execute_time, task_name='write')
    print_records_count(records_count=rowcount, task_name='write')
    
    if return_time:
        return rowcount, execute_time
    else:
        return rowcount

def read_data_csv(file, separator=',', quoting= 'MINIMAL', compression='infer', 
                    header='infer', columns=None, encoding='utf-8', on_error='ignore', 
                    return_time=False):
    """
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
    
    Parameters
    ----------    
    file : str
    separator : str
    index : bool
    compression : {'infer', 'gzip', 'bz2', 'zip', 'xz', None}, default 'infer'
    quoting : {'ALL', MINIMAL', 'NONNUMERIC', 'NONE'}, default 'MINIMAL'
    header : int, list of int, None, default 'infer'
    columns : list(str)
    encoding : {'utf-8', 'utf-16'}, default 'utf-8'
    on_error : {None, 'ignore', 'raise'}, default 'ignore'
    return_time : bool
     
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
                                compression=compression, header=header, names=columns,
                                encoding=encoding)  
        execute_time = timer() - start_time
        rowcount = len(DataFrame.index)
    except:
        execute_time = None
        DataFrame = pd.DataFrame()
        print(traceback.format_exc())
        rowcount = None    
    
    print_execute_time(execute_time, task_name='read')
    print_records_count(records_count=rowcount, task_name='read') 

    if return_time:
        return DataFrame, execute_time
    else:
        return DataFrame

def write_data_csv(DataFrame, file, separator=',', index=False, quoting='ALL', 
                    header='infer', columns=None, encoding='utf-8', compression='infer', 
                    chunksize=None, on_error='ignore', return_time=False):
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
    header : bool or list of str, default True
    columns : list(str)
    encoding : {'utf-8', 'utf-16'}, default 'utf-8'
    chunksize : int, default None
    on_error : {None, 'ignore', 'raise'}, default 'ignore'
    return_time : bool
    
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
                         header=header, names=columns, quoting=quoting, compression=compression, 
                         chunksize=chunksize)
        execute_time = timer() - start_time
        rowcount = len(DataFrame.index)
    except:
        execute_time = 0
        print(traceback.format_exc())
        rowcount = None

    print_execute_time(execute_time, task_name='write')
    print_records_count(records_count=rowcount, task_name='write') 
        
    if return_time:
        return rowcount, execute_time
    else:
        return rowcount

def set_bcp_path(bcp_path, temp_folder=None, bcp_exe_file='bcp.exe'):
    """
    Parameters
    ----------
    bcp_path : str
    temp_folder : str, default None
    bcp_exe_file : str, default 'bcp.exe'
    
    Returns
    -------
    None
    
    """
    try:
        bcp_exe_file = bcp_exe_file
        os.environ['BCP_PATH'] = bcp_path
        if temp_folder==None:
            temp_folder = os.path.join(bcp_path, 'temp') 
        os.environ['BCP_TEMP'] = temp_folder
        os.environ['BCP_APP_COMMAND'] = os.path.join(bcp_path, bcp_exe_file)
    except:
        print('ERROR in setting environment variables:\n{}'.format(traceback.format_exc()))
        
def bcp_exe(args, decode_output=True):
    # Sourse: https://docs.microsoft.com/en-us/sql/tools/bcp-utility
    """
    Base Function
    
    Parameters
    ----------
    args : list(str)
    decode_output : bool, default True
    
    Returns
    -------
    output : str
    error : str
    
    """
    output = ''
    error = ''
    
    command = os.environ['BCP_APP_COMMAND']
    command_parts = [command] + args
    try:
        # https://docs.python.org/3.6/library/subprocess.html
        p = subprocess.Popen(command_parts, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = p.communicate()
        if decode_output:
            output = output.decode()
        error = error.decode()
    except:
        print('Error in running BCP\n{}'.format(traceback.format_exc()))
    finally:
        try:
            p.kill()
        except:
            pass
    
    return output, error

    
def bcp(method, server, database, auth, schema=None, table=None, query=None, hints=None, data_file=None, 
                        batch_size=None, unicode=True, code_page=None, table_lock=False, 
                        field_term=',', row_term='\r\n', quoting='ALL', return_type=None):
    """
    Parameters
    ----------
    method : {'IN', 'OUT', 'QUERYOUT'}
    query : str
        SQL SELECT query
    server : str
        Database Server
    database : str
        Database
    auth :  dict
        e.g. auth = {'type':'user', 'user':'user', 'password':'password'} for username password authentication
             auth = {'type':'machine', 'user':None, 'password':None} for machine authentication
    schema : str 
    table : str
    data_file : str
    batch_size : int, defualt None
    unicode : bool, default True
    code_page: { None, 'ACP', 'OEM', 'RAW', <code_page>}
    table_lock : bool, default False
    return_type : {None, 'frame', 'file', json'}
       
    Returns
    -------
    DataFrame : pd.DataFrame
    
    """    

    method = 'OUT'
    
    if data_file==None:
        return None
    
    if query==None and table!=None:
        source = '"{}.{}.{}"'.format(database,schema,table)
    elif query!=None and table==None:
        source = '"{}"'.format(query), 
        
    args = [source, method, '"{}"'.format(data_file),
            '-S', '"{}"'.format(server), '-d', '"{}"'.format(database)]        

    if batch_size!=None:
        args = args + ['-b', batch_size] 
        
    if unicode:
        args = args + ['-w'] 
    else:
        args = args + ['-c'] 

    if code_page!=None:
        args = args + ['-C', '"{}"'.format(code_page)] 

    if table_lock:
        args = args + ['TABLOCK'] 
        
    if auth['type']=='machine':
        args = args + ['-T']
    elif auth['type']=='user':
        args = args + ['-U', '"{}"'.format(auth['user']), '-P', '"{}"'.format(auth['password']) ]

    if batch_size!=None:
        args = args + ['-b', batch_size] 
        
    if row_term != None:
        args = args + ['-r', '"{}"'.format(row_term)] 

    if field_term != None:
        args = args + ['-t', '"{}"'.format(field_term)] 
        
    output, error = bcp_exe(args, decode_output=True)
    
    print(error)
    
    return output

def read_data_mssql_bcp(query=None, server=None, database=None, auth=None, chunksize=None, 
                        on_error='ignore', return_time=False, save_file=None, params=None):
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
        e.g. auth = {'type':'user', 'user':'user', 'password':'password'} for username password authentication
             auth = {'type':'machine', 'uid':None, 'pwd':None} for machine authentication
    chunksize : int, default None
    on_error : {None, 'ignore', 'raise'}, default 'ignore'
    return_time : bool
    save_file : str, default None
    params : dict(), default None [Not implemented]
    
    Returns
    -------
    DataFrame : pandas.DataFrame
    """  
    
    method = 'OUT',
    code_page = None
    hints = None
    unicode = True
    field_term = ','
    row_term = '\r\n'
    quoting = 'ALL'
    table_lock = False
    batch_size = chunksize
    data_file = out_file
    quoting = 'ALL'
    temp_folder = os.environ['BCP_TEMP']
    
    if save_file==None:
        data_file = os.path.join(temp_folder, '{}{}{}'.format('bcp', str(uuid.uuid4()), '.csv')) 
    else:
        data_file = os.path.join(temp_folder, '{}'.format(save_file)) 
    
    try:
        start_time = timer()                
        output = bcp(method=method, server=server, database=database, auth=auth, query=query, 
                     data_file=data_file, batch_size=batch_size, unicode=unicode, code_page=code_page, 
                     table_lock=table_lock,  field_term=field_term, row_term=row_term, quoting=quoting, 
                     return_type=return_type)
        
        if save_file==None:
            DataFrame = read_data_csv(data_file, separator=field_term, quoting=quoting, on_error=on_error, return_time=return_time)
            
        execute_time = timer() - start_time
    except:
        print('Data read error !')
        execute_time = None
        DataFrame = pd.DataFrame()
    
    if save_file==None:
        if return_time:
            return DataFrame, execute_time
        else:
            return DataFrame
    else:
        if return_time:
            return execute_time
        else:
            return None            
    
def write_data_mssql_bcp(DataFrame=None, input_file=None, server=None, database=None, schema=None, table=None, index=False, 
                     dtypes=None, if_exists='fail', auth=None, insertion_method=None, chunksize=None,
                      on_error='ignore', return_time=False, params=None, table_lock=False):
    """
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_sql.html
    
    Parameters
    ----------
    DataFrame : pandas.DataFrame
        DataFrame
    input_file : str, defualt None
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
        e.g. auth = {'type':'user', 'user':'user', 'password':'password'} for username password authentication
             auth = {'type':'machine', 'uid':None, 'pwd':None} for machine authentication
    insertion_method : {None, 'multi', callable}
        see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_sql.html
    driver : str, default 'ODBC Driver 13 for SQL Server'
        E.g.: 'ODBC Driver 13 for SQL Server', 'ODBC Driver 17 for SQL Server', 'SQL Server'
    on_error : {None, 'ignore', 'raise'}, default 'ignore'
    return_time : bool
    table_lock : bool, defualt False
    
    Returns
    -------
    None
    """ 
    
    method = 'IN',
    code_page = None
    hints = None
    unicode = True
    field_term = ','
    row_term = '\r\n'
    quoting = 'ALL'
    table_lock = table_lock
    batch_size = chunksize
    data_file = input_file
    quoting = 'ALL'
    temp_folder = os.environ['BCP_TEMP']
    header_row=True

    if DataFrame==None and input_file!=None:
        data_file = input_file
    elif DataFrame!=None and input_file==None:
        data_file = os.path.join(temp_folder, '{}{}{}'.format('bcp', str(uuid.uuid4()), '.csv')) 
        write_data_csv(DataFrame, file=data_file, separator=field_term, index=False, quoting=quoting, 
                       chunksize=chunksize, on_error=on_error, return_time=return_time)
    else:
        print('Input error !')
        return None
        
    try:
        start_time = timer()                
        output = bcp(method=method, server=server, database=database, auth=auth, query=query, 
                     data_file=data_file, batch_size=batch_size, unicode=unicode, code_page=code_page, 
                     table_lock=table_lock,  field_term=field_term, row_term=row_term, quoting=quoting, 
                     return_type=return_type)
            
        execute_time = timer() - start_time
    except:
        print('Data read error !')
        execute_time = None
        DataFrame = pd.DataFrame()

def get_database_list(server, auth=None, user_database_only=True, dbms='mssql'):
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
    DBList = read_data_mssql(query=query, server=server, database='master', auth=auth, params=None)
    
    return DBList
    
def get_database_usage_report(server, database, auth=None, schema=None, table=None, user_tables_only=True, dbms='mssql', unit='KB'):
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
        
        DBUsageReport = read_data_mssql(query=query, server=server, database=database, auth=auth, params=None)
        
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