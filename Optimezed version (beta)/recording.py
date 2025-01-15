import tm
import csv
import platform
import os

so = platform.system()
if so == 'Linux':
    import fcntl
    import time
    import errno


@tm.measure_time
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


@tm.measure_time
def record_evolution(log_name, round, parameters, island_number, highest_values, fitness_evolution, alphabet, best_individual, island_start, island_end, island_duration, current_generation, algo_option):
    values_of_parameters = ''                                                                                           
    for i in range(len(parameters)):                                                                                    
        values_of_parameters = values_of_parameters + str(parameters[i]) + '	'                                       
    highest_values_values = ''                                                                                          
    for i in range(len(highest_values)):                                                                                
        highest_values_values = highest_values_values + str(highest_values[i]) + '	'                                   
    fields2a = [str(round) + '	' + str(island_number) + '	' + values_of_parameters + str(current_generation) + '	' + str(island_start) + '	' + str(island_end) + '	' + str(island_duration) + '	' + highest_values_values + algo_option + '	' + str(alphabet) + '	' + str(best_individual) + '	' + log_name + '	' + str(fitness_evolution)]     
    fields2b = [str(round) + '	' + str(island_number) + '	' + values_of_parameters + str(current_generation) + '	' + str(island_start) + '	' + str(island_end) + '	' + str(island_duration) + '	' + highest_values_values + algo_option]

    log_name = log_name.replace("\\", "-").replace("/", "-")

    if so == 'Windows':
        with open('output-files/' 'Log' + log_name + 'output-spreadsheet-complete.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel', delimiter=',')
            writer.writerow(fields2a)
            csvfile.close()
        with open('output-files/' 'Log' + log_name + 'output-spreadsheet-short.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel', delimiter=',')
            writer.writerow(fields2b)
            csvfile.close()
    elif so == 'Linux':
        write_to_locked_file('output-files/Log' + log_name + 'output-spreadsheet-complete.csv', fields2a)
        write_to_locked_file('output-files/Log' + log_name + 'output-spreadsheet-short.csv', fields2b)
    return


@tm.measure_time
def write_locked_bestone(file_name, data):
    fileToOpen = open(file_name, 'w')
    while True:
        try:
            fcntl.flock(fileToOpen, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for line in data:
                fileToOpen.write(line + '\n')
            fcntl.flock(fileToOpen, fcntl.LOCK_UN)
            fileToOpen.close()
            break
        except IOError as e:
            if e.errno != errno.EAGAIN:
                raise
            else:
                time.sleep(0.01)


@tm.measure_time
def record_bestone(island_number, highest_hm, generation, bestone_file, island_duration):
    if not os.path.exists(bestone_file):
        if so == 'Linux':
            write_locked_bestone(bestone_file, [str(island_number), str(highest_hm), str(generation), str(island_duration)])
        else:
            with open(bestone_file, 'w') as bestone:
                bestone.write(str(island_number) + '\n')
                bestone.write(str(highest_hm) + '\n')
                bestone.write(str(generation) + '\n')
                bestone.write(str(island_duration) + '\n')
                bestone.close()
    else:
        with open(bestone_file, 'r') as bestone:
            best_island = bestone.readline().strip()
            best_highest_hm = bestone.readline().strip()
            generation_best = bestone.readline().strip()
            total_duration = bestone.readline().strip()
            bestone.close()

        if highest_hm > float(best_highest_hm):
            if so == 'Linux':
                write_locked_bestone(bestone_file, [str(island_number), str(highest_hm), str(generation), str(island_duration)])
            else:
                with open(bestone_file, 'w') as bestone:
                    bestone.write(str(island_number) + '\n')
                    bestone.write(str(highest_hm) + '\n')
                    bestone.write(str(generation) + '\n')
                    bestone.write(str(island_duration) + '\n')
                    bestone.close()
        else:
            if so == 'Linux':
                write_locked_bestone(bestone_file, [str(best_island), str(best_highest_hm), str(generation_best), str(island_duration)])
            else:
                with open(bestone_file, 'w') as bestone:
                    bestone.write(str(best_island) + '\n')
                    bestone.write(str(best_highest_hm) + '\n')
                    bestone.write(str(generation_best) + '\n')
                    bestone.write(str(island_duration) + '\n')
                    bestone.close()
    return

# @tm.measure_time
# def write_to_locked_file_tm(file_name, data):
#     return write_to_locked_file(file_name, data)
#
# @tm.measure_time
# def record_evolution_tm(log_name, round, parameters, island_number, highest_values, fitness_evolution, alphabet, best_individual, island_start, island_end, island_duration, current_generation, algo_option):
#     return record_evolution(log_name, round, parameters, island_number, highest_values, fitness_evolution, alphabet, best_individual, island_start, island_end, island_duration, current_generation, algo_option)
#
# @tm.measure_time
# def write_locked_bestone(file_name, data):
#     return write_locked_bestone(file_name, data)
#
# @tm.measure_time
# def record_bestone(island_number, highest_hm, generation, bestone_file, island_duration):
#     return record_bestone(island_number, highest_hm, generation, bestone_file, island_duration)