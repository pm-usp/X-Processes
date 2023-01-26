from pm4py.objects.petri.petrinet import PetriNet, Marking
from pm4py.objects.petri import utils
from pm4py.objects.petri.exporter import exporter as pnml_exporter
from pm4py.visualization.petrinet import visualizer as pn_visualizer
from pm4py.algo.analysis.woflan import algorithm as woflan

import copy
import numpy

import os
#os.environ['PATH'] = os.environ['PATH'] + ';' + os.environ['CONDA_PREFIX'] + r'\Library\bin\graphviz'


def create_petri_net(cromossome, alphabet):
    net = PetriNet('petri_net')

    transitions = []

    silent_transition = PetriNet.Transition('silent-begin')
    transitions.append(silent_transition)
    net.transitions.add(silent_transition)

    for i in range(1, len(cromossome) - 2):
        transitions.append(PetriNet.Transition(alphabet[i], alphabet[i]))
        net.transitions.add(transitions[i])

    silent_transition = PetriNet.Transition('silent-end')
    transitions.append(silent_transition)
    net.transitions.add(silent_transition)

    source = PetriNet.Place('source')
    sink = PetriNet.Place('sink')
    net.places.add(source)
    net.places.add(sink)
    utils.add_arc_from_to(source, transitions[0], net)
    utils.add_arc_from_to(transitions[-1], sink, net)

    initial_marking = Marking()
    initial_marking[source] = 1
    final_marking = Marking()
    final_marking[sink] = 1

    input_places = copy.deepcopy(cromossome)
    output_places = copy.deepcopy(cromossome)
    input_place_counter = 0

    for i in range(len(cromossome)):
        input_places[-1][i] = 0
        input_places[i][-1] = 0
        output_places[-1][i] = 0
        output_places[i][-1] = 0

    last_column = numpy.sum(input_places, axis=1)
    last_row = numpy.sum(input_places, axis=0)

    for i in range(len(cromossome)):
        input_places[-1][i] = -1
        input_places[i][-1] = last_column[i]
        output_places[-1][i] = last_row[i]
        output_places[i][-1] = -1

    for i in range(len(cromossome) - 1):
        for j in range(len(cromossome) - 1):
            input_places[i][j] = None
            output_places[i][j] = None

    output_place = [None] * len(cromossome[0])
    for i in range(len(cromossome) - 2):
        input_place = None
        for j in range(1, len(cromossome) - 1):
            if cromossome[i][j] == 1:
                if (cromossome[i][-1] == 1) and (cromossome[-1][j] == 1 or (cromossome[-1][j] == 0 and output_places[-1][j] == 1)):
                    input_places[i][j] = PetriNet.Place('pi' + str(input_place_counter))
                    input_place_counter += 1
                elif (((cromossome[i][-1] == 0) and (cromossome[-1][j] == 1 or (cromossome[-1][j] == 0 and output_places[-1][j] == 1))) or (cromossome[i][-1] == 0 and cromossome[-1][j] == 1 and input_places[i][-1] > 1 and output_places[-1][j] > 1)) or (cromossome[i][-1] == 0 and cromossome[-1][j] == 0 and input_places[i][-1] > 1 and output_places[-1][j] > 1):
                    if input_place is None:
                        input_places[i][j] = PetriNet.Place('pi' + str(input_place_counter))
                        input_place_counter += 1
                        input_place = input_places[i][j]
                    else:
                        input_places[i][j] = input_place
                    if cromossome[i][-1] == 0 and cromossome[-1][j] == 0 and input_places[i][-1] > 1 and output_places[-1][j] > 1:
                        if output_place[j] is None:
                            output_places[i][j] = PetriNet.Place('pi' + str(input_place_counter))
                            output_place[j] = output_places[i][j]
                        else:
                            output_places[i][j] = output_place[j]
                        net.places.add(output_places[i][j])
                        input_place_counter += 1
                elif (((cromossome[i][-1] == 1 or cromossome[i][-1] == 0) and input_places[i][-1] == 1) and (cromossome[-1][j] == 0 and output_places[-1][j] > 1)) or (cromossome[i][-1] == 1 and cromossome[-1][j] == 0 and input_places[i][-1] > 1 and output_places[-1][j] > 1):
                    if output_place[j] is None:
                        input_places[i][j] = PetriNet.Place('pi' + str(input_place_counter))
                        input_place_counter += 1
                        output_place[j] = input_places[i][j]
                    else:
                        input_places[i][j] = output_place[j]
                net.places.add(input_places[i][j])

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

    for i in range(len(cromossome) - 2):
        for j in range(1, len(cromossome) - 1):
            if input_places[i][j] is not None:
                utils.add_arc_from_to(transitions[i], input_places[i][j], net)
                if (cromossome[i][-1] == 0) or (cromossome[i][-1] == 0 and cromossome[-1][j] == 1 and input_places[i][-1] > 1 and output_places[-1][j] > 1):
                    break

    silent_counter = 0
    place_to_transition_arcs = []
    for j in range(1, len(cromossome) - 1):
        for i in range(len(cromossome) - 2):
            if output_places[i][j] is not None:
                if (cromossome[-1][j] == 1 or (cromossome[-1][j] == 0 and output_places[-1][j] == 1)) and (cromossome[i][-1] == 1 or (cromossome[i][-1] == 0 and input_places[i][-1] == 1)):
                    if place_to_transition_arcs.count([input_places[i][j], transitions[j]]) == 0:
                        utils.add_arc_from_to(input_places[i][j], transitions[j], net)
                        place_to_transition_arcs.append([input_places[i][j], transitions[j]])
                elif (((cromossome[-1][j] == 1 or cromossome[-1][j] == 0) and output_places[-1][j] == 1) and (cromossome[i][-1] == 0 and input_places[i][-1] > 1)) or ((cromossome[-1][j] == 0 and output_places[-1][j] > 1) and ((cromossome[i][-1] == 0 or cromossome[i][-1] == 1) and input_places[i][-1] == 1)) or (cromossome[i][-1] == 1 and cromossome[-1][j] == 0 and input_places[i][-1] > 1 and output_places[-1][j] > 1) or (cromossome[i][-1] == 0 and cromossome[-1][j] == 1 and input_places[i][-1] > 1 and output_places[-1][j] > 1):
                    if place_to_transition_arcs.count([output_places[i][j], transitions[j]]) == 0:
                        utils.add_arc_from_to(output_places[i][j], transitions[j], net)
                        place_to_transition_arcs.append([output_places[i][j], transitions[j]])
                elif cromossome[i][-1] == 0 and cromossome[-1][j] == 0 and input_places[i][-1] > 1 and output_places[-1][j] > 1:
                    silent_transition = PetriNet.Transition('silent' + str(silent_counter))
                    net.transitions.add(silent_transition)
                    if place_to_transition_arcs.count([input_places[i][j], silent_transition]) == 0:
                        utils.add_arc_from_to(input_places[i][j], silent_transition, net)
                        place_to_transition_arcs.append([input_places[i][j], silent_transition])
                        utils.add_arc_from_to(silent_transition, output_places[i][j], net)
                        silent_counter += 1
                    if place_to_transition_arcs.count([output_places[i][j], transitions[j]]) == 0:
                        utils.add_arc_from_to(output_places[i][j], transitions[j], net)
                        place_to_transition_arcs.append([output_places[i][j], transitions[j]])
    return net, initial_marking, final_marking


def create_and_show_pn(cromossome, alphabet, island, generation, log_name, round):
    petrinet, initial_marking, final_marking = create_petri_net(cromossome, alphabet)
    pnml_exporter.apply(petrinet, initial_marking, 'petri-nets/petri-net-' + str(log_name) + '-' + str(round) + '-' + str(island) + '-' + str(generation) + '.pnml', final_marking)

    # gviz = pn_visualizer.apply(petrinet, initial_marking, final_marking)
    # pn_visualizer.view(gviz)
    #
    # parameters = {pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: 'svg'}
    # gviz = pn_visualizer.apply(petrinet, initial_marking, final_marking, parameters=parameters)
    # pn_visualizer.view(gviz)
    #
    # parameters = {pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: 'svg'}
    # gviz = pn_visualizer.apply(petrinet, initial_marking, final_marking, parameters=parameters)
    # pn_visualizer.save(gviz, 'petri-nets/petri-net.svg')

    return


def is_sound(cromossome, alphabet):
    petrinet, initial_marking, final_marking = create_petri_net(cromossome, alphabet)
    parameters = {woflan.Parameters.RETURN_ASAP_WHEN_NOT_SOUND: True, woflan.Parameters.PRINT_DIAGNOSTICS: False, woflan.Parameters.RETURN_DIAGNOSTICS: False}
    res = woflan.apply(petrinet, initial_marking, final_marking, parameters=parameters)
    return res
