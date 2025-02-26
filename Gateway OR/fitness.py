import petrinets as pn
import pm4py
import hashlib
import decorators

@decorators.measure_time
def hash_cromossomo_fitness(cromossomo):
    cromossomo_str = str(cromossomo.tolist())
    return hashlib.sha256(cromossomo_str.encode()).hexdigest()

@decorators.measure_time
def evaluate_population(population, alphabet, xes_log, fitness_weight, precision_weight, generalization_weight, simplicity_weight, log_name, round, island, generation, cache_fitness, cache_petri_net, fitness_lock, petri_net_lock):
    evaluation_values = []
    evaluation_sum = 0
    for i, individual in enumerate(population):
        individual_evaluation = evaluate_individual(individual, alphabet, xes_log, fitness_weight, precision_weight, generalization_weight, simplicity_weight, i, log_name, round, island, generation, cache_fitness, cache_petri_net, fitness_lock, petri_net_lock)
        evaluation_values.append(individual_evaluation)
        evaluation_sum += individual_evaluation[0]
    return [evaluation_sum, evaluation_values]

@decorators.measure_time
def evaluate_individual(cromossome, alphabet, xes_log, fitness_weight, precision_weight, generalization_weight, simplicity_weight, i, log_name, round, island, generation, cache_fitness, cache_petri_net, fitness_lock, petri_net_lock):
    cromossomo_hash = hash_cromossomo_fitness(cromossome)
    cached_value = cache_fitness.get(cromossomo_hash)
    if isinstance(cached_value, tuple) and len(cached_value) == 4:
        fitness, precision, generaliz, simplic = cached_value
        if (isinstance(fitness, dict) and 'log_fitness' in fitness and isinstance(precision, (int, float)) and isinstance(generaliz, (int, float)) and isinstance(simplic, (int, float))):
            if fitness['log_fitness'] == 0 or precision == 0 or generaliz == 0 or simplic == 0:
                return 0, fitness['log_fitness'], precision, generaliz, simplic, i, 0
            f_score = (fitness_weight + precision_weight + generalization_weight + simplicity_weight) / ((fitness_weight / fitness['log_fitness']) + (precision_weight / precision) + (generalization_weight / generaliz) + (simplicity_weight / simplic))
            return f_score, fitness['log_fitness'], precision, generaliz, simplic, i, 0
    petrinet, initial_marking, final_marking = pn.create_petri_net(cromossome, alphabet, log_name, round, island, generation, cache_petri_net, petri_net_lock)
    fitness = pm4py.fitness_alignments(xes_log, petrinet, initial_marking, final_marking)
    precision = pm4py.precision_alignments(xes_log, petrinet, initial_marking, final_marking)
    generaliz = pm4py.generalization_tbr(xes_log, petrinet, initial_marking, final_marking)
    simplic = pm4py.simplicity_petri_net(petrinet, initial_marking, final_marking, variant='arc_degree')
    if (isinstance(fitness, dict) and 'log_fitness' in fitness and isinstance(precision, (int, float)) and isinstance(generaliz, (int, float)) and isinstance(simplic, (int, float))):
        if cromossomo_hash not in cache_fitness:
            cache_fitness[cromossomo_hash] = (fitness, precision, generaliz, simplic)
    if fitness['log_fitness'] == 0 or precision == 0 or generaliz == 0 or simplic == 0:
        return 0, fitness['log_fitness'], precision, generaliz, simplic, i, 0
    f_score = (fitness_weight + precision_weight + generalization_weight + simplicity_weight) / ((fitness_weight / fitness['log_fitness']) + (precision_weight / precision) + (generalization_weight / generaliz) + (simplicity_weight / simplic))
    return f_score, fitness['log_fitness'], precision, generaliz, simplic, i, 0