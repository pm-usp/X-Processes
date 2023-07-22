import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
import re

def is_NaN(num):
    return num != num                                                                                                   

def import_log(inputlog):
    xes_log = xes_importer.apply(inputlog)
    name_inputlog = re.search(r"[^//]*$", inputlog).group(0)
    print('Log name: ', name_inputlog)
    data1 = log_converter.apply(xes_log, parameters=None, variant=log_converter.Variants.TO_DATA_FRAME)
    data2 = data1.groupby('case:concept:name')['concept:name'].apply(list).apply(pd.Series).reset_index()                             
    print('Number of cases in raw log:', len(data2))
    data2.drop('case:concept:name', axis=1, inplace=True)                                                                         
    data2.drop_duplicates(keep='first', inplace=True)                                                                   
    print('Number of traces in pre-processed log:', len(data2))
    list1 = data2.values.tolist()                                                                                       
    list2 = []                                                                                                          
    for raw in list1:
        list2_partial = []                                                                                              
        for col in raw:                                                                                                
            if not is_NaN(col):                                                                                         
                list2_partial.append(col)                                                                               
        list2.append(list2_partial)                                                                                     
    list3 = []                                                                                                          
    for raw in list2:                                                                                                  
        list3_partial = []                                                                                              
        list3_partial.append(raw[0])                                                                                   
        if (len(raw) > 1):                                                                                             
            list3_partial.append(raw[1])                                                                               
        for col in range(2, len(raw)):                                                                                 
            if (raw[col] != raw[col - 1]) or (raw[col] != raw[col - 2]):                                            
                list3_partial.append(raw[col])                                                                         
        list3.append(list3_partial)                                                                                     
    list4 = []                                                                                                          
    for num in list3:                                                                                                   
        if num not in list4:                                                                                            
            list4.append(num)                                                                                           
    log = list4                                                                                                         
    print('Number of traces in pre-processed log, excluding two or more sequences of the same task:', len(list4))       
    return log, name_inputlog, xes_log
