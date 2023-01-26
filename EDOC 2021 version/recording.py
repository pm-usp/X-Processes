import csv

def record_evolution(log_name, round, parameters, island_number, highest_values, fitness_evolution, alphabet, best_individual, island_start, island_end, island_duration, current_generation):     #?
    values_of_parameters = ''                                                                                           #?
    for i in range(len(parameters)):                                                                                    #?
        values_of_parameters = values_of_parameters + str(parameters[i]) + '	'                                       #?
    highest_values_values = ''                                                                                          #?
    for i in range(len(highest_values)):                                                                                #?
        highest_values_values = highest_values_values + str(highest_values[i]) + '	'                                   #?
    fields2a = [str(round) + '	' + str(island_number) + '	' + values_of_parameters + str(current_generation) + '	' + str(island_start) + '	' + str(island_end) + '	' + str(island_duration) + '	' + highest_values_values + str(alphabet) + '	' + str(best_individual) + '	' + log_name + '	' + str(fitness_evolution)]     #?
    fields2b = [str(round) + '	' + str(island_number) + '	' + values_of_parameters + str(current_generation) + '	' + str(island_start) + '	' + str(island_end) + '	' + str(island_duration) + '	' + highest_values_values]     #?
    with open('output-files/output-spreadsheet-complete-' + log_name + '.csv', 'a', newline='') as csvfile:               #?
        writer = csv.writer(csvfile, dialect='excel', delimiter=',')                                                    #?
        writer.writerow(fields2a)                                                                                       #?
        csvfile.close()                                                                                                 #?
    with open('output-files/output-spreadsheet-short-' + log_name + '.csv', 'a', newline='') as csvfile:                  #?
        writer = csv.writer(csvfile, dialect='excel', delimiter=',')                                                    #?
        writer.writerow(fields2b)                                                                                       #?
        csvfile.close()                                                                                                 #?
    return                                                                                                              #?
