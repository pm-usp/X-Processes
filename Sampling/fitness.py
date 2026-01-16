import petrinets as pn
import pm4py
import hashlib
import log
import math
from numba import njit
import datetime as dt
import numpy as np
import metrics.simplicity.algorithm as simplicity_calculator


def hash_cromossomo_fitness(cromossomo):
    arr = np.ascontiguousarray(np.asarray(cromossomo, dtype=np.int64))
    h = hashlib.sha256()
    h.update(arr.tobytes(order="C"))
    h.update(str(arr.shape).encode("utf-8"))
    h.update(arr.dtype.str.encode("utf-8"))
    return h.hexdigest()


def hash_alphabet(alphabet):
    h = hashlib.sha256()
    for s in alphabet:
        h.update(str(s).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def hash_log_id(sampled_xes_log_file_path: str, file_size: int):
    h = hashlib.sha256()
    h.update(sampled_xes_log_file_path.encode("utf-8"))
    h.update(b"\0")
    h.update(str(int(file_size)).encode("utf-8"))
    return h.hexdigest()


def evaluate_population(population, alphabet, fitness_weight, precision_weight, generalization_weight, simplicity_weight, simplicity_criterion_methods, input_log_name, cache_fitness, cache_petri_net, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log, filetimestamp):
    basename = input_log_name.replace("\\", "-").replace("/", "-").replace(":", "-").replace(" ", "-").replace("input-logs-", "").replace(".xes", "").replace(".gz", "")
    sampled_xes_log_file_path = f'input-logs/log-sampling/sampled-{basename}-{filetimestamp}.xes'
    cached_sampled_xes_log, last_file_size_sampled_xes_log = log.read_log_safely('sampled', sampled_xes_log_file_path, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log)
    xes_log = cached_sampled_xes_log
    alphabet_id = hash_alphabet(alphabet)
    log_id = hash_log_id(sampled_xes_log_file_path, last_file_size_sampled_xes_log or 0)
    evaluation_values = []
    evaluation_sum = 0
    for i, individual in enumerate(population):
        individual_evaluation = evaluate_individual(individual, alphabet, fitness_weight, precision_weight, generalization_weight, simplicity_weight, simplicity_criterion_methods, i, xes_log, cache_fitness, cache_petri_net, log_id, alphabet_id)
        ind_eval_fixed = (float(individual_evaluation[0]), float(individual_evaluation[1]), float(individual_evaluation[2]), float(individual_evaluation[3]), float(individual_evaluation[4]), int(individual_evaluation[5]), int(individual_evaluation[6]))
        evaluation_values.append(ind_eval_fixed)
        evaluation_sum += ind_eval_fixed[0]
    evaluated_population = [evaluation_sum, evaluation_values]
    return evaluated_population, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log


def evaluate_individual(cromossome, alphabet, fitness_weight, precision_weight, generalization_weight, simplicity_weight, simplicity_criterion_methods, i, xes_log, cache_fitness, cache_petri_net, log_id, alphabet_id):
    cromossomo_hash = hash_cromossomo_fitness(cromossome)
    cache_key = (cromossomo_hash, log_id, alphabet_id)
    cached_value = cache_fitness.get(cache_key)
    if isinstance(cached_value, tuple) and len(cached_value) == 4:
        fitness, precision, generaliz, simplic = cached_value
        if (isinstance(fitness, dict) and 'log_fitness' in fitness and isinstance(precision, (int, float)) and isinstance(generaliz, (int, float)) and isinstance(simplic, (int, float))):
            if fitness['log_fitness'] == 0 or precision == 0 or generaliz == 0 or simplic == 0:
                return 0, fitness['log_fitness'], precision, generaliz, simplic, i, 0
            #harmonic_mean = (fitness_weight + precision_weight + generalization_weight + simplicity_weight) / ((fitness_weight / fitness['log_fitness']) + (precision_weight / precision) + (generalization_weight / generaliz) + (simplicity_weight / simplic))
            power_mean = power_mean_4(fitness["log_fitness"], precision, generaliz, simplic, fitness_weight, precision_weight, generalization_weight, simplicity_weight, -0.5)
            return power_mean, fitness['log_fitness'], precision, generaliz, simplic, i, 0
    petrinet, initial_marking, final_marking, bpmn = pn.create_petri_net_and_bpmn(cromossome, alphabet, cache_petri_net)
    fitness = pm4py.fitness_alignments(xes_log, petrinet, initial_marking, final_marking)
    # aligned_traces = pm4py.conformance_diagnostics_alignments(xes_log, petrinet, initial_marking, final_marking)
    # from pm4py.algo.evaluation.replay_fitness.variants import alignment_based
    # fitness = alignment_based.evaluate(aligned_traces)
    precision = pm4py.precision_alignments(xes_log, petrinet, initial_marking, final_marking)
    generaliz = pm4py.generalization_tbr(xes_log, petrinet, initial_marking, final_marking)
    simplic, each_s = calculate_simplicity(petrinet, initial_marking, final_marking, bpmn, xes_log, simplicity_criterion_methods)
    if (isinstance(fitness, dict) and 'log_fitness' in fitness and isinstance(precision, (int, float)) and isinstance(generaliz, (int, float)) and isinstance(simplic, (int, float))):
        if cache_key not in cache_fitness:
            cache_fitness[cache_key] = (fitness, precision, generaliz, simplic)
    if fitness['log_fitness'] == 0 or precision == 0 or generaliz == 0 or simplic == 0:
        return 0, fitness['log_fitness'], precision, generaliz, simplic, i, 0
    #harmonic_mean = (fitness_weight + precision_weight + generalization_weight + simplicity_weight) / ((fitness_weight / fitness['log_fitness']) + (precision_weight / precision) + (generalization_weight / generaliz) + (simplicity_weight / simplic))
    # power_mean, p: -1 (harmonic_mean), 0 (geometric_mean), +1 (aritmetic_mean)
    power_mean = power_mean_4(fitness["log_fitness"], precision, generaliz, simplic, fitness_weight, precision_weight, generalization_weight, simplicity_weight, -0.5)
    return power_mean, fitness['log_fitness'], precision, generaliz, simplic, i, 0


def _calculate_single_simplicity(petrinet, initial_marking, final_marking, bpmn, xes_log, method: str) -> float:
    if method == 'arc_degree':
        return pm4py.simplicity_petri_net(petrinet, initial_marking, final_marking, variant='arc_degree')
    if method == 'structuredness':
        return simplicity_calculator.structuredness_java(petrinet, initial_marking, final_marking)
    if method == 'simple_structural_appropriateness':
        return simplicity_calculator.simple_structural_appropriateness(petrinet)
    if method == 'advanced_structural_appropriateness':
        return simplicity_calculator.advanced_structural_appropriateness(petrinet, initial_marking, final_marking)
    if method == 'structural_simplicity_by_edge_count':
        return simplicity_calculator.ssedc(petrinet, xes_log)
    if method == 'structural_simplicity_by_entitiy_count':
        return simplicity_calculator.ssenc(petrinet, xes_log)
    if method == 'etm_simplicity_petri_net':
        return simplicity_calculator.etm_simplicity_petri_net(petrinet, xes_log)
    if method == 'df_structural_coverage':
        return simplicity_calculator.df_structural_coverage(petrinet, xes_log)
    if method == 'sequentiality_ratio':
        return simplicity_calculator.sequentiality_ratio(bpmn)
    if method == 'cross_connectivity':
        return simplicity_calculator.cross_connectivity(bpmn)
    if method == 'flow_hierarchy':
        return simplicity_calculator.flow_hierarchy(bpmn)
    if method == 'separability_ratio':
        return simplicity_calculator.separability_java(petrinet, initial_marking, final_marking)
    if method == 'global_efficiency':
        return simplicity_calculator.global_efficiency(bpmn)
    if method == 'behavioral_specificity_q0':
        return simplicity_calculator.behavioral_specificity_q0(petrinet, initial_marking, final_marking)
    if method == 'behavioral_specificity_q1':
        return simplicity_calculator.behavioral_specificity_q1(petrinet, initial_marking, final_marking)
    if method == 'behavioral_specificity_q2':
        return simplicity_calculator.behavioral_specificity_q2(petrinet, initial_marking, final_marking)
    if method == 'cyclicity':
        return 1 - simplicity_calculator.cyclicity(bpmn)
    if method == 'density':
        return 1 - simplicity_calculator.density(bpmn)
    if method == 'connector_heterogeneity':
        return 1 - simplicity_calculator.connector_heterogeneity(bpmn)
    if method == 'coupling':
        return 1 - simplicity_calculator.coupling(bpmn)
    if method == 'alignment_complexity':
        return 1 - simplicity_calculator.alignment_complexity(petrinet, initial_marking, final_marking, xes_log)
    if method == 'network_density':
        return 1 - simplicity_calculator.network_density(bpmn)
    raise ValueError(f"Unknown simplicity method: {method}")


def calculate_simplicity(petrinet, initial_marking, final_marking, bpmn, xes_log, simplicity_criteria: dict[str, float]) -> tuple[float, list[float]]:
    if not simplicity_criteria:
        return 0.0, []
    weighted_sum = 0.0
    total_weight = 0.0
    each_s = []
    for method, w in simplicity_criteria.items():
        if w <= 0:
            continue
        try:
            s = _calculate_single_simplicity(petrinet, initial_marking, final_marking, bpmn, xes_log, method)
        except Exception as e:
            print(f"[WARN] Simplicity '{method}' failed: {e}", flush=True)
            continue
        weighted_sum += s * w
        total_weight += w
        each_s.append(s)
    if total_weight == 0:
        return 0.0, each_s
    return weighted_sum / total_weight, each_s


@njit(cache=True)
def power_mean_4(fitness: float, precision: float, generaliz: float, simplic: float, fitness_weight: float, precision_weight: float, generalization_weight: float, simplicity_weight: float, p: float, p_zero_tol: float = 1e-9, eps: float = 1e-15) -> float:
    wsum = fitness_weight + precision_weight + generalization_weight + simplicity_weight
    if wsum <= 0:
        raise ValueError("Soma dos pesos deve ser > 0.")
    fitness = max(fitness, eps)
    precision = max(precision, eps)
    generaliz = max(generaliz, eps)
    simplic = max(simplic, eps)
    if abs(p) < p_zero_tol:
        return math.exp((fitness_weight * math.log(fitness) + precision_weight * math.log(precision) + generalization_weight * math.log(generaliz) + simplicity_weight * math.log(simplic)) / wsum)
    num = (fitness_weight * (fitness ** p) + precision_weight * (precision ** p) + generalization_weight * (generaliz ** p) + simplicity_weight * (simplic ** p))
    return (num / wsum) ** (1.0 / p)
