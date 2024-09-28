import csv
import platform
import os

so = platform.system()
if so == 'Linux':
    import fcntl
    import time
    import errno

def record_evolution(log_name, round, parameters, island_number, highest_values, fitness_evolution, alphabet, best_individual, island_start, island_end, island_duration, current_generation, algo_option):     
    values_of_parameters = ''                                                                                           
    for i in range(len(parameters)):                                                                                    
        values_of_parameters = values_of_parameters + str(parameters[i]) + '	'                                       
    highest_values_values = ''                                                                                          
    for i in range(len(highest_values)):                                                                                
        highest_values_values = highest_values_values + str(highest_values[i]) + '	'                                   
    fields2a = [str(round) + '	' + str(island_number) + '	' + values_of_parameters + str(current_generation) + '	' + str(island_start) + '	' + str(island_end) + '	' + str(island_duration) + '	' + highest_values_values + algo_option + '	' + str(alphabet) + '	' + str(best_individual) + '	' + log_name + '	' + str(fitness_evolution)]     
    fields2b = [str(round) + '	' + str(island_number) + '	' + values_of_parameters + str(current_generation) + '	' + str(island_start) + '	' + str(island_end) + '	' + str(island_duration) + '	' + highest_values_values + algo_option]

    log_name = log_name.replace("\\", "-")

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
        fileToOpen = open('output-files/' 'Log' + log_name + 'output-spreadsheet-complete.csv', 'a')
        while True:
            try:
                fcntl.flock(fileToOpen, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except IOError as e:
                if e.errno != errno.EAGAIN:
                    raise
                else:
                    time.sleep(0.01)
        fcntl.flock(fileToOpen, fcntl.LOCK_UN)
        fileToOpen = open('output-files/' 'Log' + log_name + 'output-spreadsheet-short.csv', 'a')
        while True:
            try:
                fcntl.flock(fileToOpen, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except IOError as e:
                if e.errno != errno.EAGAIN:
                    raise
                else:
                    time.sleep(0.01)
        fcntl.flock(fileToOpen, fcntl.LOCK_UN)
    return


def record_bestone(island_number, highest_hm):
    if not os.path.exists('petri-nets/test.txt'):
        bestone = open('petri-nets/test.txt', 'w')
        bestone.close()

        with open('petri-nets/test.txt', 'w') as bestone:
            bestone.write(str(island_number) + '\n')
            bestone.write(str(highest_hm) + '\n')
            bestone.close()
    else:
        with open('petri-nets/test.txt', 'r') as bestone:
            best_island_number = bestone.readline().strip()
            best_highest_hm = bestone.readline().strip()
            bestone.close()

        if (highest_hm > float(best_highest_hm)):
            with open('petri-nets/test.txt', 'w') as bestone:
                bestone.write(str(island_number) + '\n')
                bestone.write(str(highest_hm) + '\n')
                bestone.close()

    with open('petri-nets/test.txt', 'r') as bestone:
        best_island_number = bestone.readline().strip()
        best_highest_hm = bestone.readline()
        bestone.close()

    print(best_island_number)
    print(best_highest_hm)

    return