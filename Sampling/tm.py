import datetime
import time
import openpyxl
from pathlib import Path

def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        duration = end - start
        timestemp = time.strftime("%H:%M:%S")
        dia_mes_ano = datetime.date.today()
        data = [func.__name__, duration, timestemp, dia_mes_ano]
        write_time_measurement(nome_arquivo, data)
        return result
    return wrapper

def create_new_workbook(file_path):
    wb = openpyxl.Workbook()
    wb.save(file_path)
    return wb

# @measure_time
def write_time_measurement(file_name, data):
    file_path = Path(file_name)
    try:
        wb = openpyxl.load_workbook(file_path)
    except:
        wb = create_new_workbook(file_path)
    sheet = wb.active
    sheet.append(data)
    wb.save(file_path)