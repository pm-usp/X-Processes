import pandas as pd
import pm4py
import re
import concurrent.futures

def remove_consecutive_repetitions(trace):
    trace_length = len(trace)
    if trace_length < 3:
        return trace
    result = [trace[0], trace[1]]
    for i in range(2, trace_length):
        if not (trace[i] == trace[i - 1] == trace[i - 2]):
            result.append(trace[i])
    return result

def import_log(inputlog):
    xes_log = pm4py.read_xes(inputlog)
    name_inputlog = re.search(r"[^//]*$", inputlog).group(0)
    data1 = pm4py.convert_to_dataframe(xes_log)
    data2 = data1.groupby('case:concept:name')['concept:name'].agg(list)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        data2_cleaned = list(executor.map(remove_consecutive_repetitions, data2))
    unique_traces = pd.Series(data2_cleaned).drop_duplicates()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        log = list(executor.map(lambda x: [event for event in x if pd.notna(event)], unique_traces))
    #print('Number of traces in pre-processed log, excluding two or more sequences of the same task:', len(log))
    return log, name_inputlog, xes_log