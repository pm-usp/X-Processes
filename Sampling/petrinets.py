import pm4py
import numpy as np
import os
import hashlib
import cloudpickle
import math
import sys
import traceback
from ast import literal_eval
import metrics.simplicity.algorithm as simplicity_calculator
import bpmn as bp
import csv
import fitness as ft
import datetime as dt
import pytz
from pm4py.objects.petri_net.utils import check_soundness as cs


def create_pn(cromossome, alphabet, island, generation, input_log_name, round, cache_petri_net):
    petrinet, initial_marking, final_marking, bpmn = create_petri_net_and_bpmn(cromossome, alphabet, cache_petri_net)
    input_log_name = input_log_name.replace("\\", "-").replace("/", "-").replace(":", "-").replace(" ", "-")
    pm4py.write_pnml(petrinet, initial_marking, final_marking, 'petri-nets/temp/' + str(input_log_name) + '-' + str(round) + '-' + str(island) + '-' + str(generation))
    return


def calculate_and_save_simplicity_metrics_columns(petrinet, initial_marking, final_marking, bpmn, input_xes_log_path, simplicity_criterion_method, output_file=f'temp/simplicity_matrix.csv'):
    metrics_data = []
    INVERT_METRICS = {'network_density', 'density', 'connector_heterogeneity', 'coupling', 'cyclicity'}
    def add_metric(name, func_call):
        try:
            val = func_call()
            val = float(val)
            if name in INVERT_METRICS:
                val = 1.0 - val
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"[{name}] {err}", file=sys.stderr, flush=True)
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            val = math.nan
        metrics_data.append((name, val))
    add_metric('arc_degree',  lambda: pm4py.simplicity_petri_net(petrinet, initial_marking, final_marking, variant='arc_degree'))
    add_metric('structuredness', lambda: simplicity_calculator.structuredness_java(petrinet, initial_marking, final_marking))
    add_metric('simple_structural_appropriateness', lambda: simplicity_calculator.simple_structural_appropriateness(petrinet))
    add_metric('advanced_structural_appropriateness', lambda: simplicity_calculator.advanced_structural_appropriateness(petrinet, initial_marking, final_marking))
    add_metric('sequentiality_ratio', lambda: simplicity_calculator.sequentiality_ratio(bpmn))
    add_metric('cross_connectivity', lambda: simplicity_calculator.cross_connectivity(bpmn))
    add_metric('flow_hierarchy', lambda: simplicity_calculator.flow_hierarchy(bpmn))
    add_metric('separability_ratio', lambda: simplicity_calculator.separability_java(petrinet, initial_marking, final_marking))
    add_metric('cyclicity', lambda: simplicity_calculator.cyclicity(bpmn))
    add_metric('density', lambda: simplicity_calculator.density(bpmn))
    add_metric('coupling', lambda: simplicity_calculator.coupling(bpmn))
    add_metric('network_density', lambda: simplicity_calculator.network_density(bpmn))
    column_header = f"{input_xes_log_path}_{simplicity_criterion_method}"
    file_exists = os.path.isfile(output_file)
    rows_to_write = []
    if not file_exists:
        rows_to_write.append(['Metric_Name', column_header])
        for name, value in metrics_data:
            rows_to_write.append([name, value])
    else:
        with open(output_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            existing_rows = list(reader)
        if len(existing_rows) != len(metrics_data) + 1:
            print(f"ERRO CRÍTICO: O arquivo {output_file} tem {len(existing_rows)} linhas, mas tentamos gravar {len(metrics_data) + 1}. A estrutura mudou?", flush=True)
            return
        existing_rows[0].append(column_header)
        for i, (name, value) in enumerate(metrics_data):
            existing_rows[i + 1].append(value)
        rows_to_write = existing_rows
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows_to_write)
    print(f"Coluna '{column_header}' adicionada ao arquivo {output_file}.", flush=True)


def write_and_show_best_pn_and_bpmn(best_cromossome, bestone_file, xes_log, input_xes_log_path, simplicity_criterion_methods, alphabet, cache_petri_net):
    bestone_file = bestone_file.replace("/", "-").replace(".", "-").replace("\\", "-").replace("-txt", "")
    literal_cromossome = literal_eval(best_cromossome)
    numpy_cromossome = np.array(literal_cromossome)
    petrinet, initial_marking, final_marking, bpmn = create_petri_net_and_bpmn(numpy_cromossome, alphabet, cache_petri_net)
    fitness = pm4py.fitness_alignments(xes_log, petrinet, initial_marking, final_marking)
    precision = pm4py.precision_alignments(xes_log, petrinet, initial_marking, final_marking)
    generaliz = pm4py.generalization_tbr(xes_log, petrinet, initial_marking, final_marking)
    petrinet, initial_marking, final_marking = pm4py.reduce_petri_net_implicit_places(petrinet, initial_marking, final_marking)
    simplic, each_s = ft.calculate_simplicity(petrinet, initial_marking, final_marking, bpmn, xes_log, simplicity_criterion_methods)
    power_mean1 = ft.power_mean_4(fitness["log_fitness"], precision, generaliz, simplic, 1, 1, 1, 1, -0.5)
    power_mean2 = ft.power_mean_4(fitness["log_fitness"], precision, generaliz, simplic, 1, 1, 1, 1, -1)
    print('Final metric values after post-processing of the best discovered model - PM:', power_mean1, 'HM:', power_mean2, 'sigle metrics:', fitness["log_fitness"], precision, generaliz, simplic, 'each simplicity metric:', each_s, flush=True)
    pm4py.write_pnml(petrinet, initial_marking, final_marking, 'petri-nets/best/' + bestone_file)
    print('Discovered process model (best found solution) saved at - pnml file:', 'petri-nets/best/' + bestone_file + '.pnml', flush=True)
    pm4py.write_bpmn(bpmn, 'petri-nets/best/' + bestone_file)  
    print('Discovered process model (best found solution) saved at - bpmn file:', 'petri-nets/best/' + bestone_file + '.bpmn', flush=True)
    pm4py.save_vis_petri_net(petrinet, initial_marking, final_marking, 'petri-nets/best/' + bestone_file + '-petri-net.png', rankdir = "LR")
    print('Discovered process model (best found solution) saved at - petri net model:', 'petri-nets/best/' + bestone_file + '-petri-net.png', flush=True)
    pm4py.save_vis_bpmn(bpmn, 'petri-nets/best/' + bestone_file + '-bpmn.png', rankdir = "LR")
    print('Discovered process model (best found solution) saved at - BPMN model:', 'petri-nets/best/' + bestone_file + '-bpmn.png', flush=True)
    calculate_and_save_simplicity_metrics_columns(petrinet, initial_marking, final_marking, bpmn, input_xes_log_path, simplicity_criterion_methods)
    return


def create_petri_net_and_bpmn(cromossome, alphabet, cache_petri_net, cromossomo_hash=None):
    if cromossomo_hash is None:
        cromossomo_hash = hash_cromossomo_petri_net(cromossome)
    raw = cache_petri_net.get(cromossomo_hash)
    if raw is not None:
        net, im, fm, bpmn_model = cloudpickle.loads(raw)
        return net, im, fm, bpmn_model
    bpmn_model = bp.generate_bpmn_graph_from_cromossome(cromossome, alphabet)
    net, im, fm = pm4py.convert_to_petri_net(bpmn_model)
    net = pm4py.reduce_petri_net_invisibles(net)
    #net, im, fm = pm4py.reduce_petri_net_implicit_places(net, im, fm)
    cache_petri_net[cromossomo_hash] = cloudpickle.dumps((net, im, fm, bpmn_model))
    return net, im, fm, bpmn_model


def is_sound(cromossome, alphabet, cache_soundness, cache_petri_net):
    cromossomo_hash = hash_cromossomo_soundness(cromossome)
    v = cache_soundness.get(cromossomo_hash)
    if isinstance(v, bool):
        return v
    petrinet, initial_marking, final_marking, bpmn = create_petri_net_and_bpmn(cromossome, alphabet, cache_petri_net, cromossomo_hash=cromossomo_hash)
    res = pm4py.check_soundness(petrinet, initial_marking, final_marking)
    if isinstance(res[0], bool) and cromossomo_hash not in cache_soundness:
        cache_soundness[cromossomo_hash] = res[0]
    return res[0]


def _hash_cromossomo_sha256_int64(cromossomo) -> str:
    a = np.asarray(cromossomo, dtype=np.int64)
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    h.update(a.tobytes())
    return h.hexdigest()


def hash_cromossomo_soundness(cromossomo):
    return _hash_cromossomo_sha256_int64(cromossomo)


def hash_cromossomo_petri_net(cromossomo):
    return _hash_cromossomo_sha256_int64(cromossomo)