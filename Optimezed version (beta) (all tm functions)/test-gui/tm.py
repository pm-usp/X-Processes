import time
import openpyxl
from pathlib import Path

nome_arquivo = 'Time_measurement.xlsx'


def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        duration = end - start
        print(f"Function {func.__name__} took {duration} seconds to run")
        data = [func.__name__, duration]
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
    print('foi escrito com sucesso')
#teste de escrita...

@measure_time
def socorro():
    return "arthur é o meu irmão e eu o amo."
    # write_time_measurement(nome_arquivo, ['funcao_teste', '1234'])

if __name__ == "__main__":
    # write_time_measurement(nome_arquivo, ["sacorro", '1.5'])
    # measure_time(socorro)
    m = socorro()
    print(m)