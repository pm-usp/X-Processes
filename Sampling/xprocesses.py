import plotting as plt
import recording as rec
import cycle as cyc
import initialpop as inp
import islands as isl
import multiprocessing as mlp
import datetime as dt
import log
from ast import literal_eval
import time
import petrinets as pn
import pytz
import argparse
import os
import operators as op
import sys
import decorators
import pm4py

pm4py.util.constants.SHOW_PROGRESS_BAR = False

sys.setrecursionlimit(1500)

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description= 'X-Processes is an automatic process model discovery method based on genetic algorithms',
                                 epilog='Example of use: python xprocesses.py -log input-logs/test.xes.gz -isl 4 -rnd 5 -gen 500 -tme 3600 -con 25 -fit 1 -prc 1 -gnl 1 -smp 1 -fix 0')

parser.add_argument('-log', required=True, type=str, help='log (default: None) | (type: .xes.gz)')
parser.add_argument('-isl', type=int, default=1, help='number of islands (default: 1) | (type: int)')
parser.add_argument('-rnd', type=int, default=1, help='number of rounds (default: 1) | (type: int)')
parser.add_argument('-gen', type=int, default=5000, help='stopping criterion - maximum number of generations (default: 5000) | (type: int)')
parser.add_argument('-tme', type=int, default=86400, help='stopping criterion - maximum execution time (default: 86400) | (type: int)')
parser.add_argument('-con', type=int, default=25, help='stopping criterion - convergence criterion, defined as the maximum consecutive number of generations with no harmonic average based-fitness improvement for the best individual (default: 25) | (type: int)')
parser.add_argument('-fit', type=float, default=1, help='fitness weight (default: 1) | (type: float)')
parser.add_argument('-prc', type=float, default=1, help='precision weight (default: 1) | (type: float)')
parser.add_argument('-gnl', type=float, default=1, help='generalization weight (default: 1) | (type: float)')
parser.add_argument('-smp', type=float, default=1, help='simplicity weight (default: 1) | (type: float)')
parser.add_argument('-fix', type=int, default=1, help='activity selection mode: 1 for fixed number of activities as in the event log, 0 for dynamic subset optimization during fitness evolution (default: 1) | (type: int)')
parser.add_argument('-spl', type=int, default=0, help='event log sampling: 1 for processing with sampling, 0 for processing with full event log (default: 0) | (type: int)')
parser.add_argument('-inc', type=float, default=0.25, help='percentage increment of event log sampling, when sampling strategy is being used (default: 0.25) | (type: float)')
parser.add_argument('-sic', type=int, default=5, help='sampling increasing criteria - convergence criterion, defined as the maximum consecutive number of generations with no harmonic average based-fitness improvement for the best individual (default: 5) | (type: int)')
parser.add_argument('-tmc', type=int, default=0, help='timeout control: 1 for enabled, 0 for disabled (default: 0) | (type: int)')

args = parser.parse_args()

input_xes_log = args.log
number_of_islands = args.isl
number_of_rounds = args.rnd
max_number_of_generations = args.gen
max_processing_time = args.tme
stop_condition = args.con
fitness_weight = args.fit
precision_weight = args.prc
generalization_weight = args.gnl
simplicity_weight = args.smp
act_fix_mode = args.fix
event_log_sampling = args.spl
if event_log_sampling:
    sampling_increment_percentage = args.inc
else:
    sampling_increment_percentage = 1
sampling_increasing_criteria = args.sic
timeout_control = bool(args.tmc)

@decorators.measure_time
def calculate_statistics(evaluated_population):
    highest_value_and_position, sorted_evaluated_population = cyc.choose_highest(evaluated_population)
    lowest_value = cyc.choose_lowest(sorted_evaluated_population)
    average_value = cyc.calculate_average(evaluated_population)
    return highest_value_and_position, lowest_value, average_value, sorted_evaluated_population

def run_round(paramenter_set, number_of_islands, round, broadcast, messenger, island_sizes, percentage_of_best_individuals_for_migration_all_islands, alphabet, input_log_name, max_number_of_generations, max_processing_time, stop_condition, bestone_file, cache_fitness, cache_soundness, cache_petri_net, cache_n_tokens, sampled_log_lock, remaining_log_lock, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log, filetimestamp, island_number):
    try:
        island_start = dt.datetime.now()
        island_params = paramenter_set[island_number + 1]
        population_size = int(island_params[0])
        migration_time = int(island_params[1])
        percentage_of_best_individuals_for_migration_per_island = float(island_params[2])
        percentage_of_individuals_for_migration_per_island = float(island_params[3])
        max_perc_of_num_tasks_for_crossover = float(island_params[4])
        crossover_probability = float(island_params[5])
        task_mutation_probability = float(island_params[6])
        gateway_mutation_probability = float(island_params[7])
        max_perc_of_num_tasks_for_task_mutation = float(island_params[8])
        max_perc_of_num_tasks_for_gateway_mutation = float(island_params[9])
        elitism_percentage = float(island_params[10])
        on_off_task_mutation_probability = float(island_params[11])
        on_off_task_mutation_probability_final = float(island_params[11])
        max_perc_of_num_tasks_for_on_off_task_mutation = float(island_params[12])
        sampled_xes_log, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log = log.sample_log(sampling_increment_percentage, input_log_name, sampled_log_lock, remaining_log_lock, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log, filetimestamp)
        df_log = log.transform_from_sampled_xes_log(sampled_xes_log)
        inp.process_log(df_log)
        island_sizes[island_number] = population_size
        percentage_of_best_individuals_for_migration_all_islands[island_number] = percentage_of_best_individuals_for_migration_per_island
        population, evaluated_population, reference_cromossome, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log = inp.initialize_population(population_size, alphabet, df_log, fitness_weight, precision_weight, generalization_weight, simplicity_weight, input_log_name, round, island_number, 0, cache_fitness, cache_soundness, cache_petri_net, cache_n_tokens, act_fix_mode, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log, filetimestamp)
        fitness_evolution = []
        highest_value_and_position, lowest_value, average_value, sorted_evaluated_population = calculate_statistics(evaluated_population)
        fitness_evolution.append([lowest_value, highest_value_and_position[0][0], average_value, 0, highest_value_and_position[0][1], highest_value_and_position[0][2], highest_value_and_position[0][3], highest_value_and_position[0][4], 0, 0, 0, 0])
        pn.create_pn(population[int(highest_value_and_position[1])], alphabet, island_number, 0 , input_log_name, round, cache_petri_net)
        island_end = dt.datetime.now()
        island_duration = island_end - island_start
        number_of_sampled_cases, number_of_remaining_cases, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log = log.calculate_numbers_of_sampled_and_remaining_cases(input_log_name, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log, filetimestamp)
        sampling_rate = number_of_sampled_cases / (number_of_sampled_cases + number_of_remaining_cases) * 100
        print('r', round, '%.5f' % highest_value_and_position[0][0], '%.5f' % highest_value_and_position[0][1], '%.5f' % highest_value_and_position[0][2], '%.5f' % highest_value_and_position[0][3], '%.5f' % highest_value_and_position[0][4], '| i', '{:>2}'.format(island_number), '| g', '{:>3}'.format('0'), '| REP', '{:>2}'.format(fitness_evolution[0][8]), '{:>2}'.format(fitness_evolution[0][3]), '{:>2}'.format(fitness_evolution[0][9]), '{:>2}'.format(fitness_evolution[0][10]), '{:>2}'.format(fitness_evolution[0][11]), '|', op.count_active_inactive_tasks(population[int(highest_value_and_position[1])]), '|', str(island_duration).split('.')[0], '|', dt.datetime.now(pytz.timezone("Brazil/East")).strftime("%H:%M:%S"), '|', '%.1f%%' % sampling_rate)
        rec.record_bestone_atomic(island_number, highest_value_and_position[0][0], 0, bestone_file, island_duration)
        isl.set_broadcast(population, sorted_evaluated_population, island_number, percentage_of_best_individuals_for_migration_per_island, broadcast)
        if isl.receive_individuals(population, island_number, sorted_evaluated_population, messenger, evaluated_population) == 1:
            (highest_value_and_position, sorted_evaluated_population) = cyc.choose_highest(evaluated_population)
        if migration_time == 1:
            island_fitness = [evaluated_population[1][i][0] for i in range(len(evaluated_population[1]))]
            isl.do_migration(population, island_number, number_of_islands, island_fitness, percentage_of_individuals_for_migration_per_island, broadcast, island_sizes, percentage_of_best_individuals_for_migration_all_islands, evaluated_population)
            isl.send_individuals(population, island_number, number_of_islands, sorted_evaluated_population, messenger)
            highest_value_and_position, sorted_evaluated_population = cyc.choose_highest(evaluated_population)
        rec.record_evolution(input_log_name, str(round), paramenter_set[island_number + 1], island_number, highest_value_and_position[0], fitness_evolution, alphabet, population[int(highest_value_and_position[1])], island_start, island_end, island_duration, 0)
        for current_generation in range(1, max_number_of_generations):
            population, evaluated_population, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log = cyc.generation(population, reference_cromossome, evaluated_population, crossover_probability, max_perc_of_num_tasks_for_crossover, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, elitism_percentage, sorted_evaluated_population, alphabet, fitness_weight, precision_weight, generalization_weight, simplicity_weight, input_log_name, round, island_number, current_generation, on_off_task_mutation_probability, max_perc_of_num_tasks_for_on_off_task_mutation, cache_fitness, cache_soundness, cache_petri_net, cache_n_tokens, act_fix_mode, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log, filetimestamp)
            highest_value_and_position, lowest_value, average_value, sorted_evaluated_population = calculate_statistics(evaluated_population)
            fitness_evolution.append([lowest_value, highest_value_and_position[0][0], average_value, 0, highest_value_and_position[0][1], highest_value_and_position[0][2], highest_value_and_position[0][3], highest_value_and_position[0][4], 0, 0, 0, 0])
            fitness_evolution[current_generation][8] = (fitness_evolution[current_generation - 1][8] + 1 if fitness_evolution[current_generation][1] == fitness_evolution[current_generation - 1][1] else -1 if fitness_evolution[current_generation][1] < fitness_evolution[current_generation - 1][1] else fitness_evolution[current_generation][8])
            fitness_evolution[current_generation][3] = (fitness_evolution[current_generation - 1][3] + 1 if fitness_evolution[current_generation][4] == fitness_evolution[current_generation - 1][4] else -1 if fitness_evolution[current_generation][4] < fitness_evolution[current_generation - 1][4] else fitness_evolution[current_generation][3])
            fitness_evolution[current_generation][9] = (fitness_evolution[current_generation - 1][9] + 1 if fitness_evolution[current_generation][5] == fitness_evolution[current_generation - 1][5] else -1 if fitness_evolution[current_generation][5] < fitness_evolution[current_generation - 1][5] else fitness_evolution[current_generation][9])
            fitness_evolution[current_generation][10] = (fitness_evolution[current_generation - 1][10] + 1 if fitness_evolution[current_generation][6] == fitness_evolution[current_generation - 1][6] else -1 if fitness_evolution[current_generation][6] < fitness_evolution[current_generation - 1][6] else fitness_evolution[current_generation][10])
            fitness_evolution[current_generation][11] = (fitness_evolution[current_generation - 1][11] + 1 if fitness_evolution[current_generation][7] == fitness_evolution[current_generation - 1][7] else -1 if fitness_evolution[current_generation][7] < fitness_evolution[current_generation - 1][7] else fitness_evolution[current_generation][11])
            pn.create_pn(population[int(highest_value_and_position[1])], alphabet, island_number, current_generation, input_log_name, round, cache_petri_net)
            island_end = dt.datetime.now()
            island_duration = island_end - island_start
            number_of_sampled_cases, number_of_remaining_cases, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log = log.calculate_numbers_of_sampled_and_remaining_cases(input_log_name, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log, filetimestamp)
            sampling_rate = number_of_sampled_cases / (number_of_sampled_cases + number_of_remaining_cases) * 100
            print('r', round, '%.5f' % highest_value_and_position[0][0], '%.5f' % highest_value_and_position[0][1], '%.5f' % highest_value_and_position[0][2], '%.5f' % highest_value_and_position[0][3], '%.5f' % highest_value_and_position[0][4], '| i', '{:>2}'.format(island_number), '| g', '{:>3}'.format(current_generation), '| REP', '{:>2}'.format(fitness_evolution[current_generation][8]), '{:>2}'.format(fitness_evolution[current_generation][3]), '{:>2}'.format(fitness_evolution[current_generation][9]), '{:>2}'.format(fitness_evolution[current_generation][10]), '{:>2}'.format(fitness_evolution[current_generation][11]), '|', op.count_active_inactive_tasks(population[int(highest_value_and_position[1])]), '|', str(island_duration).split('.')[0], '|', dt.datetime.now(pytz.timezone("Brazil/East")).strftime("%H:%M:%S"), '|', '%.1f%%' % sampling_rate)
            rec.record_bestone_atomic(island_number, highest_value_and_position[0][0], current_generation, bestone_file, island_duration)
            isl.set_broadcast(population, sorted_evaluated_population, island_number, percentage_of_best_individuals_for_migration_per_island, broadcast)
            if ((fitness_evolution[current_generation][8] >= sampling_increasing_criteria)) or (island_duration > dt.timedelta(seconds=max_processing_time)):
                if number_of_remaining_cases > 0:
                    sampled_xes_log, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log = log.sample_log(sampling_increment_percentage, input_log_name, sampled_log_lock, remaining_log_lock, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log, filetimestamp)
                if ((fitness_evolution[current_generation][8] >= stop_condition)) or (island_duration > dt.timedelta(seconds=max_processing_time)):
                    break
            if isl.receive_individuals(population, island_number, sorted_evaluated_population, messenger, evaluated_population) == 1:
                (highest_value_and_position, sorted_evaluated_population) = cyc.choose_highest(evaluated_population)
            if (current_generation > 0) and (current_generation % migration_time == 0):
                island_fitness = []
                for i in range(len(evaluated_population[1])):
                    island_fitness.append(evaluated_population[1][i][0])
                isl.do_migration(population, island_number, number_of_islands, island_fitness, percentage_of_individuals_for_migration_per_island, broadcast, island_sizes, percentage_of_best_individuals_for_migration_all_islands, evaluated_population)
                isl.send_individuals(population, island_number, number_of_islands, sorted_evaluated_population, messenger)
                (highest_value_and_position, sorted_evaluated_population) = cyc.choose_highest(evaluated_population)
            rec.record_evolution(input_log_name, str(round), paramenter_set[island_number + 1], island_number, highest_value_and_position[0], fitness_evolution, alphabet, population[int(highest_value_and_position[1])], island_start, island_end, island_duration, current_generation)
            if (fitness_evolution[current_generation][8] >= (stop_condition * 2 / 10)):
                on_off_task_mutation_probability = on_off_task_mutation_probability_final
        plt.plot_evolution_per_island(fitness_evolution, input_log_name, str(round), island_number)
        with open('output-graphs/plotting/plotting_{0}.txt'.format(island_number), 'r') as plott:
            previous_plotting = [literal_eval(line) for line in isl.nonblank_lines(plott)]
        previous_plotting.extend(fitness_evolution)
        with open('output-graphs/plotting/plotting_{0}.txt'.format(island_number), 'w') as plott:
            plott.writelines(f"{str(entry)}\n" for entry in previous_plotting)
        rec.record_evolution(input_log_name, str(round), paramenter_set[island_number + 1], island_number, highest_value_and_position[0], fitness_evolution, alphabet, population[int(highest_value_and_position[1])], island_start, island_end, island_duration, current_generation)
    except Exception as e:
        import traceback
        error_message = traceback.format_exc()
        print("Exceção capturada em run_round (ilha {}):\n{}".format(island_number, error_message))
        return {"error": error_message}


def monitor_with_timeout(processes, start_times, max_processing_time, number_of_islands, finished_times, min_timeout=1800, check_interval=10, base_factor=2.5, shrink_factor=1.5, completion_exponent=0.9, time_exponent=0.9):
    # base_factor ==>> Tolerância inicial razoável (não agressivo) ==> Se parecer ainda permissivo demais no começo, diminua base_factor
    # shrink_factor ==>> Tolerância final (agressivo moderado) ==> Se parecer apertado demais no fim, aumente shrink_factor
    # completion_exponent ==>> Suaviza decaimento por ilhas (suave, sublinear) 🔸 completion_exponent: Controla quanto o progresso das ilhas afeta o timeout. ➔ Menor que 1: As primeiras ilhas finalizando já fazem o timeout começar a cair perceptivelmente. ➔ Maior que 1: As primeiras ilhas não afetam quase nada, o timeout só despenca quando 70-80% das ilhas já terminaram.
    # time_exponent ==>> Suaviza decaimento por tempo (suave, sublinear) 🔸 time_exponent: Controla quanto o tempo absoluto afeta o timeout. ➔ Menor que 1: O tempo já começa a apertar cedo na execução, mesmo longe do limite de 24h. ➔ Maior que 1: O tempo só começa a afetar o timeout significativamente perto do fim da janela (ex.: 80-90% do tempo total).
    round_start_time = time.time()
    last_finished = None
    while True:
        current_time = time.time()
        total_elapsed_time = current_time - round_start_time
        time_ratio = min(total_elapsed_time / max_processing_time, 1.0)
        alive = [p.is_alive() for p in processes]
        num_finished = sum(1 for t in finished_times if t is not None)
        completion_ratio = num_finished / number_of_islands if number_of_islands > 0 else 0
        for i, proc in enumerate(processes):
            if not alive[i] and finished_times[i] is None:
                finished_times[i] = current_time - start_times[i]
                last_finished = finished_times[i]
        for i, proc in enumerate(processes):
            if alive[i]:
                elapsed = current_time - start_times[i]
                if last_finished is not None:
                    decay_by_completion = completion_ratio ** completion_exponent
                    decay_by_time = time_ratio ** time_exponent
                    decay_factor = (decay_by_completion + decay_by_time) / 2
                    dynamic_factor = base_factor - (base_factor - shrink_factor) * decay_factor
                    timeout_limit = last_finished * dynamic_factor
                    effective_timeout = max(timeout_limit, min_timeout)
                    effective_timeout = min(effective_timeout, (max_processing_time + 300))
                    if elapsed > effective_timeout:
                        print(f"[TIMEOUT] Island {i} exceeded {effective_timeout:.1f}s. Current time: {elapsed:.1f}s. Finishing.")
                        proc.terminate()
                        proc.join()
        if not any(alive):
            break
        time.sleep(check_interval)


if __name__ == '__main__':
    sys.setrecursionlimit(10000)
    print(dt.datetime.now(pytz.timezone("Brazil/East")).strftime("%H:%M:%S"))
    paramenter_set = []
    with open('input-parameters/input-parameters.csv', 'r') as parameters:
        for line in isl.nonblank_lines(parameters):
            paramenter_set.append(line.split(';'))
    full_df_log, full_xes_log, input_log_name = log.import_log(input_xes_log)
    print(dt.datetime.now(pytz.timezone("Brazil/East")).strftime("%H:%M:%S"))
    alphabet = inp.create_alphabet(full_df_log)
    for round in range(0, number_of_rounds):
        filetimestamp = str(dt.datetime.now()).replace(" ", "-").replace(":", "-").replace(" ", "-").replace(".", "-")
        basename = input_log_name.replace("\\", "-").replace("/", "-").replace(":", "-").replace(" ", "-").replace("input-logs-", "").replace(".xes", "").replace(".gz", "")
        sampled_xes_log_file_path = f'input-logs/log-sampling/sampled-{basename}-{filetimestamp}.xes'
        remaining_xes_log_file_path = f'input-logs/log-sampling/remaining-{basename}-{filetimestamp}.xes'
        if os.path.exists(sampled_xes_log_file_path):
            os.remove(sampled_xes_log_file_path)
        if os.path.exists(remaining_xes_log_file_path):
            os.remove(remaining_xes_log_file_path)
        pm4py.write_xes(full_xes_log, remaining_xes_log_file_path)
        bestone_file = 'petri-nets/' + input_log_name.replace("\\", "-").replace("/", "-").replace(":", "-").replace(" ", "-") + filetimestamp + '.txt'
        if os.path.exists(bestone_file):
            os.remove(bestone_file)
        island_ids = list(range(number_of_islands))
        isl.create_plotting_files(island_ids, number_of_islands)
        m = mlp.Manager()
        cache_fitness = m.dict()
        cache_soundness = m.dict()
        cache_petri_net = m.dict()
        cache_n_tokens = m.dict()
        sampled_log_lock = m.Lock()
        remaining_log_lock = m.Lock()
        cached_sampled_xes_log = None
        cached_remaining_xes_log = None
        last_file_size_sampled_xes_log = None
        last_file_size_remaining_xes_log = None
        broadcast = m.list([[] for _ in range(number_of_islands)] + [1])
        messenger = m.list([[[], [], []] for _ in range(number_of_islands)])
        island_sizes = m.list([0] * number_of_islands)
        percentage_of_best_individuals_for_migration_all_islands = m.list([0] * number_of_islands)
        finished_times = [None] * number_of_islands
        last_finished = None
        start_times = [time.time() for _ in range(number_of_islands)]
        processes = []
        for island_id in island_ids:
            proc = mlp.Process(target=run_round, args=(paramenter_set, number_of_islands, round, broadcast, messenger, island_sizes, percentage_of_best_individuals_for_migration_all_islands, alphabet, input_log_name, max_number_of_generations, max_processing_time, stop_condition, bestone_file, cache_fitness, cache_soundness, cache_petri_net, cache_n_tokens, sampled_log_lock, remaining_log_lock, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log, filetimestamp, island_id))
            processes.append(proc)
            proc.start()
        if timeout_control:
            monitor_with_timeout(processes, start_times, max_processing_time, number_of_islands, finished_times)
        else:
            for proc in processes:
                proc.join()
        plt.plot_evolution_integrated(input_log_name, str(round), number_of_islands)
        with open(bestone_file, 'r') as bestone:
            best_island_number = bestone.readline().strip()
            best_highest_hm = bestone.readline().strip()
            generation_best = bestone.readline().strip()
            duration_total = bestone.readline().strip()
            bestone.close()
        print('Best island: ', best_island_number)
        print('Highest HM: ', best_highest_hm)
        print('Generation: ', generation_best)
        print('Duration: ', duration_total)
        pn.write_and_show_best_pn(best_island_number, input_log_name, round, generation_best, bestone_file, full_xes_log, fitness_weight, precision_weight, generalization_weight, simplicity_weight)
        os.remove(bestone_file)