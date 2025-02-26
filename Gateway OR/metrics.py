import pm4py
import decorators

@decorators.measure_time
def calculate_metrics(log, pnml_file, fitness_weight, precision_weight, generalization_weight, simplicity_weight):
    petrinet, initial_marking, final_marking = pm4py.read_pnml(pnml_file)

    # print(pm4py.check_soundness(petrinet, initial_marking, final_marking, 1))
    fitness = pm4py.fitness_alignments(log, petrinet, initial_marking, final_marking)
    precision = 1 #pm4py.precision_alignments(log, petrinet, initial_marking, final_marking)
    generaliz =  1 #pm4py.generalization_tbr(log, petrinet, initial_marking, final_marking)
    simplic =  1 #pm4py.simplicity_petri_net(petrinet, initial_marking, final_marking, variant='arc_degree')
    if fitness['log_fitness'] == 0 or precision == 0 or generaliz == 0 or simplic == 0:
        return 0, fitness['log_fitness'], precision, generaliz, simplic
    f_score = (fitness_weight + precision_weight + generalization_weight + simplicity_weight) / ((fitness_weight / fitness['log_fitness']) + (precision_weight / precision) + (generalization_weight / generaliz) + (simplicity_weight / simplic))
    return f_score, fitness['log_fitness'], precision, generaliz, simplic

log_xes = 'input-logs/BPIC14-PreProcessed-Filtered2.xes.gz'
log = pm4py.read_xes(log_xes)
pnml_file = 'petri-nets/temp/BPIC14-PreProcessed-Filtered.xes.gz-0-0-0-2024-11-09-22-54-04-774143.pnml'
print(calculate_metrics(log, pnml_file, 1, 1, 1, 1))
