import pandas as pd
import pm4py
import re
import concurrent.futures
import decorators
import os
import time
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import gzip

@decorators.measure_time
def remove_consecutive_repetitions(trace):
    trace_length = len(trace)
    if trace_length < 3:
        return trace
    result = [trace[0], trace[1]]
    for i in range(2, trace_length):
        if not (trace[i] == trace[i - 1] == trace[i - 2]):
            result.append(trace[i])
    return result

def is_gzip_file(filename):
    with open(filename, 'rb') as f:
        return f.read(2) == b'\x1f\x8b'

def has_timestamp_in_xes(xes_file):
    open_fn = gzip.open if is_gzip_file(xes_file) else open
    with open_fn(xes_file, 'rt', encoding="utf-8-sig") as f:
        for line in f:
            if 'key="time:timestamp"' in line:
                return True
    return False

def add_fictitious_timestamps_file(input_xes, output_xes, start_time="2024-01-01T00:00:00.000+00:00", increment_seconds=1):
    tree = ET.parse(input_xes)
    root = tree.getroot()
    current_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    for trace in root.findall("trace"):
        for event in trace.findall("event"):
            if not any(attr.get("key") == "time:timestamp" for attr in event):
                timestamp_elem = ET.Element("date")
                timestamp_elem.set("key", "time:timestamp")
                timestamp_elem.set("value", current_time.isoformat())
                event.append(timestamp_elem)
                current_time += timedelta(seconds=increment_seconds)

    tree.write(output_xes, encoding="UTF-8", xml_declaration=True)
    print(f"Timestamps fictícios adicionados no arquivo temporário: {output_xes}", flush=True)

@decorators.measure_time
def import_log(input_xes_log):
    if not has_timestamp_in_xes(input_xes_log):
        print("Nenhum timestamp encontrado no XES. Gerando timestamps fictícios...", flush=True)
        # Cria arquivo temporário com timestamps
        temp_fd, temp_xes_path = tempfile.mkstemp(suffix=".xes")
        os.close(temp_fd)  # Fecha o descritor de arquivo
        add_fictitious_timestamps_file(input_xes_log, temp_xes_path)
        xes_log_path = temp_xes_path
    else:
        xes_log_path = input_xes_log

    # Carrega o log (original ou temporário)
    xes_log = pm4py.read_xes(xes_log_path)
    input_log_name = re.search(r"[^//]*$", input_xes_log).group(0)
    data1 = pm4py.convert_to_dataframe(xes_log)
    data2 = data1.groupby('case:concept:name')['concept:name'].agg(list)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        data2_cleaned = list(executor.map(remove_consecutive_repetitions, data2))
    unique_traces = pd.Series(data2_cleaned).drop_duplicates()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        listOfLists_log = list(executor.map(lambda x: [event for event in x if pd.notna(event)], unique_traces))
    return listOfLists_log, xes_log, input_log_name

@decorators.measure_time
def sample_log(perc, input_log_name, sampled_log_lock, remaining_log_lock, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log, filetimestamp):
    # sampled_xes_log_file_path = 'input-logs/log-sampling/sampled-' + str(input_log_name.replace("\\", "-").replace("/", "-").replace(":", "-").replace(" ", "-")).replace("input-logs-", "").replace(".xes", "").replace(".gz", "") + '.xes'
    # remaining_xes_log_file_path = 'input-logs/log-sampling/remaining-' + str(input_log_name.replace("\\", "-").replace("/", "-").replace(":", "-").replace(" ", "-")).replace("input-logs-", "").replace(".xes", "").replace(".gz", "") + '.xes'
    basename = input_log_name.replace("\\", "-").replace("/", "-").replace(":", "-").replace(" ", "-").replace("input-logs-", "").replace(".xes", "").replace(".gz", "")
    sampled_xes_log_file_path = f'input-logs/log-sampling/sampled-{basename}-{filetimestamp}.xes'
    remaining_xes_log_file_path = f'input-logs/log-sampling/remaining-{basename}-{filetimestamp}.xes'
    if os.path.exists(sampled_xes_log_file_path):
        cached_sampled_xes_log, last_file_size_sampled_xes_log = read_log_safely('sampled', sampled_xes_log_file_path, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log)
        current_sampled_xes_log = cached_sampled_xes_log
    else:
        current_sampled_xes_log = []
    cached_remaining_xes_log, last_file_size_remaining_xes_log = read_log_safely('remaining', remaining_xes_log_file_path, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log)
    current_remaining_xes_log = cached_remaining_xes_log
    if len(current_remaining_xes_log) > 0:
        data1a = pm4py.convert_to_dataframe(current_remaining_xes_log)
    else:
        return current_sampled_xes_log, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log
    data1b = data1a.groupby('case:concept:name')['concept:name'].agg(list)
    total_number_of_remaining_cases = len(data1b)
    if len(current_sampled_xes_log) > 0:
        data2a = pm4py.convert_to_dataframe(current_sampled_xes_log)
        data2b = data2a.groupby('case:concept:name')['concept:name'].agg(list)
        total_number_of_sampled_cases = len(data2b)
    else:
        total_number_of_sampled_cases = 0
    sampling_number_of_cases = int((total_number_of_sampled_cases + total_number_of_remaining_cases) * perc) + 1
    if sampling_number_of_cases > total_number_of_remaining_cases:
        sampling_number_of_cases = total_number_of_remaining_cases
    new_sampled_xes_log = pm4py.sample_cases(current_remaining_xes_log, sampling_number_of_cases, case_id_key='case:concept:name')
    data3 = pm4py.convert_to_dataframe(new_sampled_xes_log)
    if len(current_sampled_xes_log) > 0:
        data4 = pm4py.convert_to_dataframe(current_sampled_xes_log)
        data5 = pd.concat([data3, data4], ignore_index=True)
        new_sampled_xes_log = pm4py.convert_to_event_log(data5)
    else:
        new_sampled_xes_log = pm4py.convert_to_event_log(data3)
    sampled_df = pm4py.convert_to_dataframe(new_sampled_xes_log)
    sampled_case_ids = set(sampled_df['case:concept:name'].unique())
    remaining_df = data1a[~data1a['case:concept:name'].isin(sampled_case_ids)]
    remaining_xes_log = pm4py.convert_to_event_log(remaining_df)
    write_log_with_lock(new_sampled_xes_log, sampled_xes_log_file_path, sampled_log_lock)
    write_log_with_lock(remaining_xes_log, remaining_xes_log_file_path, remaining_log_lock)
    return new_sampled_xes_log, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log

@decorators.measure_time
def transform_from_sampled_xes_log(sampled_xes_log):
    data1 = pm4py.convert_to_dataframe(sampled_xes_log)
    data2 = data1.groupby('case:concept:name')['concept:name'].agg(list)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        data2_cleaned = list(executor.map(remove_consecutive_repetitions, data2))
    unique_traces = pd.Series(data2_cleaned).drop_duplicates()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        sampled_listOfLists_log = list(executor.map(lambda x: [event for event in x if pd.notna(event)], unique_traces))
    return sampled_listOfLists_log

@decorators.measure_time
def calculate_numbers_of_sampled_and_remaining_cases(input_log_name, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log, filetimestamp):
    # sampled_xes_log_file_path = 'input-logs/log-sampling/sampled-' + str(input_log_name.replace("\\", "-").replace("/", "-").replace(":", "-").replace(" ", "-")).replace("input-logs-", "").replace(".xes", "").replace(".gz", "") + '.xes'
    # remaining_xes_log_file_path = 'input-logs/log-sampling/remaining-' + str(input_log_name.replace("\\", "-").replace("/", "-").replace(":", "-").replace(" ", "-")).replace("input-logs-", "").replace(".xes", "").replace(".gz", "") + '.xes'
    basename = input_log_name.replace("\\", "-").replace("/", "-").replace(":", "-").replace(" ", "-").replace("input-logs-", "").replace(".xes", "").replace(".gz", "")
    sampled_xes_log_file_path = f'input-logs/log-sampling/sampled-{basename}-{filetimestamp}.xes'
    remaining_xes_log_file_path = f'input-logs/log-sampling/remaining-{basename}-{filetimestamp}.xes'
    cached_sampled_xes_log, last_file_size_sampled_xes_log = read_log_safely('sampled', sampled_xes_log_file_path, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log)
    current_sampled_xes_log = cached_sampled_xes_log
    cached_remaining_xes_log, last_file_size_remaining_xes_log = read_log_safely('remaining', remaining_xes_log_file_path, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log)
    current_remaining_xes_log = cached_remaining_xes_log
    if len(current_remaining_xes_log) > 0:
        data1a = pm4py.convert_to_dataframe(current_remaining_xes_log)
        data1b = data1a.groupby('case:concept:name')['concept:name'].agg(list)
        total_number_of_remaining_cases = len(data1b)
    else:
        total_number_of_remaining_cases = 0
    data2a = pm4py.convert_to_dataframe(current_sampled_xes_log)
    data2b = data2a.groupby('case:concept:name')['concept:name'].agg(list)
    total_number_of_sampled_cases = len(data2b)
    return total_number_of_sampled_cases, total_number_of_remaining_cases, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log

@decorators.measure_time
def write_log_with_lock(xes_log, xes_log_file_path, log_lock):
    with log_lock:
        while True:
            try:
                pm4py.write_xes(xes_log, xes_log_file_path)
                break
            except Exception as e:
                time.sleep(0.01)

@decorators.measure_time
def read_log_safely(log_type, xes_log_file_path, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log):

    @decorators.measure_time
    def read_xes_log_safely(cached_xes_log, last_file_size):
        current_file_size = os.stat(xes_log_file_path, follow_symlinks=True).st_size
        if current_file_size == last_file_size:
            return cached_xes_log, last_file_size
        while True:
            file_size_before = os.stat(xes_log_file_path, follow_symlinks=True).st_size
            try:
                new_xes_log = pm4py.read_xes(xes_log_file_path)
                file_size_after = os.stat(xes_log_file_path, follow_symlinks=True).st_size
                if file_size_before == file_size_after:
                    return new_xes_log, file_size_after
            except Exception:
                pass
            time.sleep(0.01)

    if log_type == 'sampled':
        return read_xes_log_safely(cached_sampled_xes_log, last_file_size_sampled_xes_log)
    else:
        return read_xes_log_safely(cached_remaining_xes_log, last_file_size_remaining_xes_log)