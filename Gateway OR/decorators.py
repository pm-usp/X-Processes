import datetime
import time
import csv
from pathlib import Path
import pytz

filename = 'output-files/time_measurement.csv'

def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        write_time_measurement(
            filename,
            [func.__name__, time.time() - start, datetime.datetime.now(pytz.timezone("Brazil/East")).strftime("%d/%m/%Y %H:%M:%S")]
        )
        return result
    return wrapper

def write_time_measurement(file_name, data):
    file_path = Path(file_name)
    file_exists = file_path.exists()
    with open(file_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Function Name", "Duration (s)", "Timestamp"])
        writer.writerow(data)