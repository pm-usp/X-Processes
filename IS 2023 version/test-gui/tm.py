# This code refers to the alternative to measure the time spend of the functions in the project
# With the purpose to discover the bottlenecks of the x-processes method :)
import csv
import time

file_name = 'Time_measurement.xlsx'

def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function {func.__name__} took {end - start} seconds to run")
        return result
    return wrapper

# This function has the purpose to write the results of the function measure_time in a Excel file
# called 'Time_measurement.xlsx' locate in the path: X-Processes/Optimezed version (beta)/test-gui
@measure_time
def write_time_measurement(id_function,timespend):
    fileToOpen = open(file_name, 'a')
    writer = csv.writer(fileToOpen, dialect='excel', delimiter=',')

    writer.writerow(id_function)
    # writer.writerow(timespend)

    fileToOpen.close()
