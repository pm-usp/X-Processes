import csv
import platform
import os
import decorators
import tempfile
import time
import uuid
import portalocker

so = platform.system()
if so == 'Linux':
    import fcntl
    import time
    import errno


@decorators.measure_time
def write_to_locked_file(file_name, data):
    fileToOpen = open(file_name, 'a')
    while True:
        try:
            fcntl.flock(fileToOpen, fcntl.LOCK_EX | fcntl.LOCK_NB)
            writer = csv.writer(fileToOpen, dialect='excel', delimiter=',')
            writer.writerow(data)
            fcntl.flock(fileToOpen, fcntl.LOCK_UN)
            fileToOpen.close()
            break
        except IOError as e:
            if e.errno != errno.EAGAIN:
                raise
            else:
                time.sleep(0.01)


@decorators.measure_time
def record_evolution(input_log_name, round, parameters, island_number, highest_values, fitness_evolution, alphabet, best_individual, island_start, island_end, island_duration, current_generation):
    values_of_parameters = ''                                                                                           
    for i in range(len(parameters)):                                                                                    
        values_of_parameters = values_of_parameters + str(parameters[i]) + '	'                                       
    highest_values_values = ''                                                                                          
    for i in range(len(highest_values)):                                                                                
        highest_values_values = highest_values_values + str(highest_values[i]) + '	'                                   
    fields2a = [str(round) + '	' + str(island_number) + '	' + values_of_parameters + str(current_generation) + '	' + str(island_start) + '	' + str(island_end) + '	' + str(island_duration) + '	' + highest_values_values + '	' + str(alphabet) + '	' + str(best_individual) + '	' + input_log_name + '	' + str(fitness_evolution)]
    fields2b = [str(round) + '	' + str(island_number) + '	' + values_of_parameters + str(current_generation) + '	' + str(island_start) + '	' + str(island_end) + '	' + str(island_duration) + '	' + highest_values_values]

    input_log_name = input_log_name.replace("\\", "-").replace("/", "-").replace(":", "-").replace(" ", "-")

    if so == 'Windows':
        with open('output-files/' 'Log' + input_log_name + 'output-spreadsheet-complete.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel', delimiter=',')
            writer.writerow(fields2a)
            csvfile.close()
        with open('output-files/' 'Log' + input_log_name + 'output-spreadsheet-short.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel', delimiter=',')
            writer.writerow(fields2b)
            csvfile.close()
    elif so == 'Linux':
        write_to_locked_file('output-files/Log' + input_log_name + 'output-spreadsheet-complete.csv', fields2a)
        write_to_locked_file('output-files/Log' + input_log_name + 'output-spreadsheet-short.csv', fields2b)
    return


@decorators.measure_time
def write_atomic_bestone(file_path, data, max_retries=100, base_wait=0.05):
    unique_id = uuid.uuid4().hex
    temp_path = f"{file_path}-tmp-{unique_id}"
    with open(temp_path, 'w') as f:
        for line in data:
            f.write(f"{line}\n")
    retries = 0
    while True:
        try:
            os.replace(temp_path, file_path)
            break
        except PermissionError as e:
            retries += 1
            if retries > max_retries:
                raise RuntimeError(f"Failed to replace {file_path} after {max_retries} retries due to file locking.") from e
            wait_time = base_wait * (2 ** (retries - 1))
            time.sleep(wait_time)
        except FileNotFoundError as e:
            raise RuntimeError(f"Temporary file {temp_path} not found. This indicates a logic error or external interference.") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error replacing {file_path}: {e}") from e


@decorators.measure_time
def read_bestone(file_path, max_retries=100, base_wait=0.05):
    retries = 0
    while True:
        try:
            with open(file_path, 'r') as file:
                portalocker.lock(file, portalocker.LOCK_SH)
                lines = [line.strip() for line in file.readlines()]
                portalocker.unlock(file)
            if len(lines) != 4:
                raise ValueError(f'File {file_path} has incorrect format. Expected 4 lines, got {len(lines)}.')
            return lines
        except FileNotFoundError as e:
            raise e
        except PermissionError as e:
            retries += 1
            if retries > max_retries:
                raise RuntimeError(f"Failed to read {file_path} after {max_retries} retries due to file locking.") from e
            wait_time = base_wait * (2 ** (retries - 1))
            time.sleep(wait_time)
        except Exception as e:
            raise RuntimeError(f"Unexpected error reading {file_path}: {e}") from e


@decorators.measure_time
def record_bestone_atomic(island_number, highest_hm, generation, bestone_file, island_duration):
    try:
        best_island, best_highest_hm, generation_best, total_duration = read_bestone(bestone_file)
        best_highest_hm = float(best_highest_hm)
    except (FileNotFoundError, ValueError):
        best_island, best_highest_hm, generation_best, total_duration = None, None, None, None
    if best_highest_hm is None or highest_hm > best_highest_hm:
        data = [str(island_number), str(highest_hm), str(generation), str(island_duration)]
    else:
        data = [str(best_island), str(best_highest_hm), str(generation_best), str(island_duration)]
    write_atomic_bestone(bestone_file, data)