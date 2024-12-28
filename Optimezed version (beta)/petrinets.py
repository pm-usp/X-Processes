import tm
import numpy as np
import pm4py
import concurrent.futures
import os
#os.environ['PATH'] = os.environ['PATH'] + ';' + os.environ['CONDA_PREFIX'] + r'\Library\bin\graphviz'

def initialize_places(input_places, output_places):
    input_places[-1, :] = 0
    input_places[:, -1] = 0
    output_places[-1, :] = 0
    output_places[:, -1] = 0

def adjust_places_sums(input_places, output_places):
    last_column = np.sum(input_places, axis=1)
    last_row = np.sum(input_places, axis=0)
    input_places[-1, :] = -1
    input_places[:, -1] = last_column
    output_places[-1, :] = last_row
    output_places[:, -1] = -1

def clear_internal_places(input_places, output_places):
    input_places[:-1, :-1] = None
    output_places[:-1, :-1] = None

def configure_input_output_places(cromossome, input_places, output_places, output_place, net):
    input_place_counter = 0
    for i in range(len(cromossome) - 2):
        input_place = None
        for j in range(1, len(cromossome) - 1):
            if cromossome[i][j] == 1:
                if (cromossome[i][-1] == 1) and (cromossome[-1][j] == 1 or (cromossome[-1][j] == 0 and output_places[-1][j] == 1)):
                    input_places[i][j] = pm4py.objects.petri_net.obj.PetriNet.Place('pi' + str(input_place_counter))
                    input_place_counter += 1
                elif (((cromossome[i][-1] == 0) and (cromossome[-1][j] == 1 or (cromossome[-1][j] == 0 and output_places[-1][j] == 1))) or (cromossome[i][-1] == 0 and cromossome[-1][j] == 1 and input_places[i][-1] > 1 and output_places[-1][j] > 1)) or (cromossome[i][-1] == 0 and cromossome[-1][j] == 0 and input_places[i][-1] > 1 and output_places[-1][j] > 1):
                    if input_place is None:
                        input_places[i][j] = pm4py.objects.petri_net.obj.PetriNet.Place('pi' + str(input_place_counter))
                        input_place_counter += 1
                        input_place = input_places[i][j]
                    else:
                        input_places[i][j] = input_place
                    if cromossome[i][-1] == 0 and cromossome[-1][j] == 0 and input_places[i][-1] > 1 and output_places[-1][j] > 1:
                        if output_place[j] is None:
                            output_places[i][j] = pm4py.objects.petri_net.obj.PetriNet.Place('pi' + str(input_place_counter))
                            output_place[j] = output_places[i][j]
                        else:
                            output_places[i][j] = output_place[j]
                        net.places.add(output_places[i][j])
                        input_place_counter += 1
                elif (((cromossome[i][-1] == 1 or cromossome[i][-1] == 0) and input_places[i][-1] == 1) and (cromossome[-1][j] == 0 and output_places[-1][j] > 1)) or (cromossome[i][-1] == 1 and cromossome[-1][j] == 0 and input_places[i][-1] > 1 and output_places[-1][j] > 1):
                    if output_place[j] is None:
                        input_places[i][j] = pm4py.objects.petri_net.obj.PetriNet.Place('pi' + str(input_place_counter))
                        input_place_counter += 1
                        output_place[j] = input_places[i][j]
                    else:
                        input_places[i][j] = output_place[j]
                net.places.add(input_places[i][j])

def configure_output_places(cromossome, input_places, output_places, net):
    for j in range(1, len(cromossome) - 1):
        input_place = None
        for i in range(len(cromossome) - 2):
            if cromossome[i][j] == 1:
                if ((cromossome[-1][j] == 1 or (cromossome[-1][j] == 0 and output_places[-1][j] == 1)) and (cromossome[i][-1] == 1 or (cromossome[i][-1] == 0 and input_places[i][-1] == 1))) or (cromossome[i][-1] == 0 and cromossome[-1][j] == 1 and input_places[i][-1] > 1 and output_places[-1][j] > 1):
                    output_places[i][j] = input_places[i][j]
                    net.places.add(output_places[i][j])
                elif (((cromossome[-1][j] == 1 or cromossome[-1][j] == 0) and output_places[-1][j] == 1) and (cromossome[i][-1] == 0 and input_places[i][-1] > 1)) or ((cromossome[-1][j] == 0 and output_places[-1][j] > 1) and ((cromossome[i][-1] == 0 or cromossome[i][-1] == 1) and input_places[i][-1] == 1)) or (cromossome[i][-1] == 1 and cromossome[-1][j] == 0 and input_places[i][-1] > 1 and output_places[-1][j] > 1):
                    if input_place is None:
                        input_place = input_places[i][j]
                    output_places[i][j] = input_place
                    net.places.add(output_places[i][j])
                    if ((cromossome[-1][j] == 0 and output_places[-1][j] > 1) and ((cromossome[i][-1] == 0 or cromossome[i][-1] == 1) and input_places[i][-1] == 1)) or (cromossome[i][-1] == 1 and cromossome[-1][j] == 0 and input_places[i][-1] > 1 and output_places[-1][j] > 1):
                        break

def add_input_to_transition_arcs(cromossome, input_places, transitions, net, output_places):
    place_to_transition_arcs = set()

    def process_arc(i, j):
        if input_places[i][j] is not None and (transitions[i], input_places[i][j]) not in place_to_transition_arcs:
            pm4py.objects.petri_net.utils.petri_utils.add_arc_from_to(transitions[i], input_places[i][j], net)
            place_to_transition_arcs.add((transitions[i], input_places[i][j]))
            if (cromossome[i][-1] == 0) or (cromossome[i][-1] == 0 and cromossome[-1][j] == 1 and input_places[i][-1] > 1 and output_places[-1][j] > 1):
                return False
        return True

    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     futures = [executor.submit(process_arc, i, j) for i in range(len(cromossome) - 2) for j in range(1, len(cromossome) - 1)]
    #     concurrent.futures.wait(futures)

    for i in range(len(cromossome) - 2):
        for j in range(1, len(cromossome) - 1):
            process_arc(i, j)

def add_output_to_transition_arcs(cromossome, input_places, output_places, transitions, net):
    silent_counter = 0
    place_to_transition_arcs = set()  # Substituir por um conjunto
    for j in range(1, len(cromossome) - 1):
        for i in range(len(cromossome) - 2):
            if output_places[i][j] is not None:
                if (cromossome[-1][j] == 1 or (cromossome[-1][j] == 0 and output_places[-1][j] == 1)) and \
                   (cromossome[i][-1] == 1 or (cromossome[i][-1] == 0 and input_places[i][-1] == 1)):
                    if (input_places[i][j], transitions[j]) not in place_to_transition_arcs:
                        pm4py.objects.petri_net.utils.petri_utils.add_arc_from_to(input_places[i][j], transitions[j], net)
                        place_to_transition_arcs.add((input_places[i][j], transitions[j]))  # Adicionar ao conjunto
                elif (((cromossome[-1][j] == 1 or cromossome[-1][j] == 0) and output_places[-1][j] == 1) and
                      (cromossome[i][-1] == 0 and input_places[i][-1] > 1)) or \
                     ((cromossome[-1][j] == 0 and output_places[-1][j] > 1) and
                      ((cromossome[i][-1] == 0 or cromossome[i][-1] == 1) and input_places[i][-1] == 1)) or \
                     (cromossome[i][-1] == 1 and cromossome[-1][j] == 0 and input_places[i][-1] > 1 and output_places[-1][j] > 1) or \
                     (cromossome[i][-1] == 0 and cromossome[-1][j] == 1 and input_places[i][-1] > 1 and output_places[-1][j] > 1):
                    if (output_places[i][j], transitions[j]) not in place_to_transition_arcs:
                        pm4py.objects.petri_net.utils.petri_utils.add_arc_from_to(output_places[i][j], transitions[j], net)
                        place_to_transition_arcs.add((output_places[i][j], transitions[j]))
                elif cromossome[i][-1] == 0 and cromossome[-1][j] == 0 and input_places[i][-1] > 1 and output_places[-1][j] > 1:
                    silent_transition = pm4py.objects.petri_net.obj.PetriNet.Transition('silent' + str(silent_counter))
                    net.transitions.add(silent_transition)
                    if (input_places[i][j], silent_transition) not in place_to_transition_arcs:
                        pm4py.objects.petri_net.utils.petri_utils.add_arc_from_to(input_places[i][j], silent_transition, net)
                        place_to_transition_arcs.add((input_places[i][j], silent_transition))
                        pm4py.objects.petri_net.utils.petri_utils.add_arc_from_to(silent_transition, output_places[i][j], net)
                        silent_counter += 1
                    if (output_places[i][j], transitions[j]) not in place_to_transition_arcs:
                        pm4py.objects.petri_net.utils.petri_utils.add_arc_from_to(output_places[i][j], transitions[j], net)
                        place_to_transition_arcs.add((output_places[i][j], transitions[j]))

def create_petri_net(cromossome, alphabet):
    net = pm4py.objects.petri_net.obj.PetriNet('petri_net')
    transitions = []
    silent_transition = pm4py.objects.petri_net.obj.PetriNet.Transition('silent-begin')
    transitions.append(silent_transition)
    net.transitions.add(silent_transition)
    for i in range(1, len(cromossome) - 2):
        transitions.append(pm4py.objects.petri_net.obj.PetriNet.Transition(alphabet[i], alphabet[i]))
        net.transitions.add(transitions[i])
    silent_transition = pm4py.objects.petri_net.obj.PetriNet.Transition('silent-end')
    transitions.append(silent_transition)
    net.transitions.add(silent_transition)
    source = pm4py.objects.petri_net.obj.PetriNet.Place('source')
    sink = pm4py.objects.petri_net.obj.PetriNet.Place('sink')
    net.places.add(source)
    net.places.add(sink)
    pm4py.objects.petri_net.utils.petri_utils.add_arc_from_to(source, transitions[0], net)
    pm4py.objects.petri_net.utils.petri_utils.add_arc_from_to(transitions[-1], sink, net)
    initial_marking = pm4py.objects.petri_net.obj.Marking()
    initial_marking[source] = 1
    final_marking = pm4py.objects.petri_net.obj.Marking()
    final_marking[sink] = 1
    input_places = np.array([row[:] for row in cromossome.tolist()], dtype=object)
    output_places = np.array([row[:] for row in cromossome.tolist()], dtype=object)
    initialize_places(input_places, output_places)
    adjust_places_sums(input_places, output_places)
    clear_internal_places(input_places, output_places)
    output_place = [None] * len(cromossome[0])
    configure_input_output_places(cromossome, input_places, output_places, output_place, net)
    configure_output_places(cromossome, input_places, output_places, net)
    add_input_to_transition_arcs(cromossome, input_places, transitions, net, output_places)
    add_output_to_transition_arcs(cromossome, input_places, output_places, transitions, net)
    net = pm4py.reduce_petri_net_invisibles(net)
    net, initial_marking, final_marking = pm4py.reduce_petri_net_implicit_places(net, initial_marking, final_marking)
    return net, initial_marking, final_marking

def create_pn(cromossome, alphabet, island, generation, log_name, round):
    petrinet, initial_marking, final_marking = create_petri_net(cromossome, alphabet)
    log_name = log_name.replace("\\", "-").replace("/", "-")
    pm4py.write_pnml(petrinet, initial_marking, final_marking, 'petri-nets/temp/' + str(log_name) + '-' + str(round) + '-' + str(island) + '-' + str(generation))
    return

def write_and_show_best_pn(best_island_number, log_name, round, generation_best, bestone_file):
    petrinet_name = 'petri-nets/temp/' + str(log_name.replace("\\", "-").replace("/", "-")) + '-' + str(round) + '-' + str(best_island_number) + '-' + str(generation_best) + '.pnml'
    petrinet, initial_marking, final_marking = pm4py.read_pnml(petrinet_name)
    bestone_file = bestone_file.replace("/", "-").replace(".", "-").replace("\\", "-")
    pm4py.write_pnml(petrinet, initial_marking, final_marking, 'petri-nets/best/' + bestone_file)
    pm4py.view_petri_net(petrinet, initial_marking, final_marking)
    return

def is_sound(cromossome, alphabet):
    petrinet, initial_marking, final_marking = create_petri_net(cromossome, alphabet)
    res = pm4py.check_soundness(petrinet, initial_marking, final_marking)
    return res[0]

@tm.measure_time
def initialize_places_tm(input_places, output_places):
    return initialize_places(input_places, output_places)

@tm.measure_time
def adjust_places_sums_tm(input_places, output_places):
    return adjust_places_sums(input_places, output_places)

@tm.measure_time
def clear_internal_places_tm(input_places, output_places):
    return clear_internal_places(input_places, output_places)

@tm.measure_time
def configure_input_output_places_tm(cromossome, input_places, output_places, output_place, net):
    return configure_input_output_places(cromossome, input_places, output_places, output_place, net)

@tm.measure_time
def configure_output_places_tm(cromossome, input_places, output_places, net):
    return configure_output_places(cromossome, input_places, output_places, net)

@tm.measure_time
def add_input_to_transition_arcs_tm(cromossome, input_places, transitions, net):
    return add_input_to_transition_arcs(cromossome, input_places, transitions, net)

@tm.measure_time
def add_output_to_transition_arcs_tm(cromossome, input_places, output_places, transitions, net):
    return add_output_to_transition_arcs(cromossome, input_places, output_places, transitions, net)
@tm.measure_time
def create_petri_net_tm(cromossome, alphabet):
    return create_petri_net(cromossome, alphabet)
@tm.measure_time
def create_pn_tm(cromossome, alphabet, island, generation, log_name, round):
    return create_pn(cromossome, alphabet, island, generation, log_name, round)
@tm.measure_time
def write_and_show_best_pn_tm(best_island_number, log_name, round, generation_best, bestone_file):
    return write_and_show_best_pn(best_island_number, log_name, round, generation_best, bestone_file)
@tm.measure_time
def is_sound_tm(cromossome, alphabet):
    return is_sound(cromossome, alphabet)