import pandas as pd
import pm4py
import re
import concurrent.futures
import os
import time
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import gzip
import unicodedata
from functools import lru_cache


@lru_cache(maxsize=50000)
def _norm_label_str(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_label(v) -> str:
    return _norm_label_str(str(v))


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


def import_log(input_xes_log_path):
    if not has_timestamp_in_xes(input_xes_log_path):
        print("Nenhum timestamp encontrado no XES. Gerando timestamps fictícios...", flush=True)
        temp_fd, temp_xes_path = tempfile.mkstemp(suffix=".xes")
        os.close(temp_fd)
        add_fictitious_timestamps_file(input_xes_log_path, temp_xes_path)
        xes_log_path = temp_xes_path
    else:
        xes_log_path = input_xes_log_path
    xes_log = pm4py.read_xes(xes_log_path)
    input_log_name = re.search(r"[^//]*$", input_xes_log_path).group(0)
    data1 = pm4py.convert_to_dataframe(xes_log)
    data2 = data1.groupby('case:concept:name')['concept:name'].agg(list)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        data2_cleaned = list(executor.map(remove_consecutive_repetitions, data2))
    unique_traces = pd.Series(data2_cleaned).drop_duplicates()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        listOfLists_log = list(executor.map(lambda x: [lab for lab in (_norm_label(event) for event in x if pd.notna(event)) if lab != ""],unique_traces))
    return listOfLists_log, xes_log, input_log_name


def _stable_sort_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["case:concept:name"]
    if "time:timestamp" in df.columns:
        df = df.copy()
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce", utc=True)
        cols.append("time:timestamp")
    if "@@index" in df.columns:
        cols.append("@@index")
    return df.sort_values(cols, kind="mergesort")


def sample_log(perc, input_xes_log_path, input_log_name, sampled_log_lock, remaining_log_lock, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log, filetimestamp):
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
    total_number_of_remaining_cases = int(data1a['case:concept:name'].nunique())
    if len(current_sampled_xes_log) > 0:
        data2a = pm4py.convert_to_dataframe(current_sampled_xes_log)
        total_number_of_sampled_cases = int(data2a['case:concept:name'].nunique())
    else:
        total_number_of_sampled_cases = 0
    sampling_number_of_cases = int((total_number_of_sampled_cases + total_number_of_remaining_cases) * perc) + 1
    if sampling_number_of_cases > total_number_of_remaining_cases:
        sampling_number_of_cases = total_number_of_remaining_cases
    if total_number_of_remaining_cases > 0 and sampling_number_of_cases == total_number_of_remaining_cases:
        full_xes_log = pm4py.read_xes(input_xes_log_path)
        empty_df = pm4py.convert_to_dataframe(full_xes_log).iloc[0:0]
        remaining_xes_log = pm4py.convert_to_event_log(empty_df)
        write_log_with_lock(full_xes_log, sampled_xes_log_file_path, sampled_log_lock)
        write_log_with_lock(remaining_xes_log, remaining_xes_log_file_path, remaining_log_lock)
        cached_sampled_xes_log = full_xes_log
        cached_remaining_xes_log = remaining_xes_log
        last_file_size_sampled_xes_log = os.stat(sampled_xes_log_file_path, follow_symlinks=True).st_size
        last_file_size_remaining_xes_log = os.stat(remaining_xes_log_file_path, follow_symlinks=True).st_size
        return full_xes_log, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log
    new_sampled_xes_log = pm4py.sample_cases(current_remaining_xes_log, sampling_number_of_cases, case_id_key='case:concept:name')
    data3 = _stable_sort_df(pm4py.convert_to_dataframe(new_sampled_xes_log))
    if len(current_sampled_xes_log) > 0:
        data4 = _stable_sort_df(pm4py.convert_to_dataframe(current_sampled_xes_log))
        data5 = pd.concat([data3, data4], ignore_index=True)
        data5 = _stable_sort_df(data5)
        new_sampled_xes_log = pm4py.convert_to_event_log(data5)
    else:
        new_sampled_xes_log = pm4py.convert_to_event_log(_stable_sort_df(data3))
    if len(current_sampled_xes_log) > 0:
        sampled_case_ids = set(data5['case:concept:name'].unique())
    else:
        sampled_case_ids = set(data3['case:concept:name'].unique())
    remaining_df = data1a[~data1a['case:concept:name'].isin(sampled_case_ids)]
    remaining_df = _stable_sort_df(remaining_df)
    remaining_xes_log = pm4py.convert_to_event_log(remaining_df)
    write_log_with_lock(new_sampled_xes_log, sampled_xes_log_file_path, sampled_log_lock)
    write_log_with_lock(remaining_xes_log, remaining_xes_log_file_path, remaining_log_lock)
    return new_sampled_xes_log, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log


def transform_from_sampled_xes_log(sampled_xes_log):
    data1 = pm4py.convert_to_dataframe(sampled_xes_log)
    data2 = data1.groupby('case:concept:name')['concept:name'].agg(list)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        data2_cleaned = list(executor.map(remove_consecutive_repetitions, data2))
    unique_traces = pd.Series(data2_cleaned).drop_duplicates()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        sampled_listOfLists_log = list(executor.map(lambda x: [lab for lab in (_norm_label(event) for event in x if pd.notna(event)) if lab != ""],unique_traces))
    return sampled_listOfLists_log


def calculate_numbers_of_sampled_and_remaining_cases(input_log_name, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log, filetimestamp):
    basename = input_log_name.replace("\\", "-").replace("/", "-").replace(":", "-").replace(" ", "-").replace("input-logs-", "").replace(".xes", "").replace(".gz", "")
    sampled_xes_log_file_path = f'input-logs/log-sampling/sampled-{basename}-{filetimestamp}.xes'
    remaining_xes_log_file_path = f'input-logs/log-sampling/remaining-{basename}-{filetimestamp}.xes'
    cached_sampled_xes_log, last_file_size_sampled_xes_log = read_log_safely('sampled', sampled_xes_log_file_path, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log)
    current_sampled_xes_log = cached_sampled_xes_log
    cached_remaining_xes_log, last_file_size_remaining_xes_log = read_log_safely('remaining', remaining_xes_log_file_path, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log)
    current_remaining_xes_log = cached_remaining_xes_log
    if len(current_remaining_xes_log) > 0:
        data1a = pm4py.convert_to_dataframe(current_remaining_xes_log)
        total_number_of_remaining_cases = int(data1a['case:concept:name'].nunique())
    else:
        total_number_of_remaining_cases = 0
    data2a = pm4py.convert_to_dataframe(current_sampled_xes_log)
    total_number_of_sampled_cases = int(data2a['case:concept:name'].nunique())
    return total_number_of_sampled_cases, total_number_of_remaining_cases, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log


def write_log_with_lock(xes_log, xes_log_file_path, log_lock):
    tmp_dir = os.path.dirname(xes_log_file_path) or "."
    with log_lock:
        while True:
            tmp_path = None
            try:
                fd, tmp_path = tempfile.mkstemp(dir=tmp_dir, suffix=".tmp.xes")
                os.close(fd)
                pm4py.write_xes(xes_log, tmp_path)
                os.replace(tmp_path, xes_log_file_path)  # atômico no mesmo filesystem
                break
            except Exception:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                time.sleep(0.01)


def read_log_safely(log_type, xes_log_file_path, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log):

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