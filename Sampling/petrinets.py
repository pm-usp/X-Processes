import pm4py
import xml.etree.ElementTree as ET
import numpy as np
import datetime as dt
import os
import hashlib
import decorators
import cloudpickle
import fitness as ft

@decorators.measure_time
def get_gateway_type(value):
    if value == 3:
        return "bpmn:inclusiveGateway"
    elif value == 2:
        return "bpmn:parallelGateway"
    elif value == 1:
        return "bpmn:exclusiveGateway"


import xml.etree.ElementTree as ET
import numpy as np


def get_gateway_type(value):
    """
    Exemplo simples. Altere conforme sua lógica real.
    value > 0 -> 'bpmn:exclusiveGateway' ou 'bpmn:parallelGateway', etc.
    """
    # Vamos supor que value == 1 => exclusiveGateway, value >= 2 => parallelGateway
    if value == 1:
        return "bpmn:exclusiveGateway"
    else:
        return "bpmn:parallelGateway"


def generate_bpmn_from_cromossome(cromossome, alphabet):
    bpmn = ET.Element(
        "bpmn:definitions",
        {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xmlns:bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL",
            "xmlns:bpmndi": "http://www.omg.org/spec/BPMN/20100524/DI",
            "xmlns:dc": "http://www.omg.org/spec/DD/20100524/DC",
            "xmlns:di": "http://www.omg.org/spec/DD/20100524/DI",
            "id": "Definitions_001",
            "targetNamespace": "http://bpmn.io/schema/bpmn"
        }
    )

    process = ET.SubElement(
        bpmn,
        "bpmn:process",
        {
            "id": "Process_001",
            "isExecutable": "false"
        }
    )

    # Start e End
    ET.SubElement(process, "bpmn:startEvent", {"id": "StartEvent_1"})
    ET.SubElement(process, "bpmn:endEvent", {"id": "EndEvent_1"})

    # --------------------------------------------------------
    # 1) Criar atividades e guardar num dicionário
    # --------------------------------------------------------
    activities = []
    for idx, activity in enumerate(alphabet):
        # Se essas condições realmente fazem sentido no seu modelo, mantenha;
        # senão, ajuste conforme necessário.
        if cromossome[idx, -1] == -1 and cromossome[-1, idx] == -1:
            continue
        task = ET.SubElement(
            process,
            "bpmn:task",
            {
                "id": f"Activity_{idx}",
                "name": activity
            }
        )
        activities.append(task)

    # --------------------------------------------------------
    # 2) Criar gateways (split e join) conforme cromossomo
    # --------------------------------------------------------
    gateways = {}

    # 2.1) split gateways (com base na última coluna ou outra lógica)
    for idx, value in enumerate(cromossome[:, -1]):
        if value > 0:
            # Cria um gateway normal
            gateway_type = get_gateway_type(value)
            gateway = ET.SubElement(
                process,
                gateway_type,
                {"id": f"Gateway_split_{idx}"}
            )
            gateways[f"split_{idx}"] = gateway

    # 2.2) join gateways (com base na última linha ou outra lógica)
    for idx, value in enumerate(cromossome[-1, :-1]):
        if value > 0:
            gateway_type = get_gateway_type(value)
            gateway = ET.SubElement(
                process,
                gateway_type,
                {"id": f"Gateway_join_{idx}"}
            )
            gateways[f"join_{idx}"] = gateway

    # 2.3) split XOR (exemplo: se a soma da linha excluindo última coluna for > 1 e a última coluna for 0)
    for idx, row in enumerate(cromossome[:-1]):
        if np.sum(row[:-1]) > 1 and row[-1] == 0:
            gateway = ET.SubElement(
                process,
                "bpmn:exclusiveGateway",
                {"id": f"Gateway_split_XOR_{idx}"}
            )
            gateways[f"split_XOR_{idx}"] = gateway

    # 2.4) join XOR (exemplo: se a soma da coluna excluindo última linha for > 1 e a última linha for 0)
    for idx in range(cromossome.shape[1] - 1):
        column = cromossome[:-1, idx]
        if np.sum(column) > 1 and cromossome[-1, idx] == 0:
            gateway = ET.SubElement(
                process,
                "bpmn:exclusiveGateway",
                {"id": f"Gateway_join_XOR_{idx}"}
            )
            gateways[f"join_XOR_{idx}"] = gateway

    # --------------------------------------------------------
    # 3) Criar sequenceFlows (AGORA SEM DUPLICAÇÕES)
    # --------------------------------------------------------
    # Percorrer cada linha (atividade fonte)
    # Determinar se tem split_XOR_{idx} ou split_{idx}
    # Em cada célula, se value == 1, criar fluxo com joinXOR, join normal ou direto
    # --------------------------------------------------------
    for idx, row in enumerate(cromossome):
        base_source_id = f"Activity_{idx}"

        # --------------------------------------------------------------------
        # 3.1) Descobrir se a fonte terá um gateway de split
        #     Se tiver split_XOR_{idx}, esse será o "source".
        #     Se não tiver, mas existir split_{idx}, será esse.
        #     Caso contrário, será a própria atividade.
        # --------------------------------------------------------------------
        if f"split_XOR_{idx}" in gateways:
            source_id = gateways[f"split_XOR_{idx}"].attrib['id']
            # Cria o fluxo da Activity -> Gateway split_XOR
            ET.SubElement(
                process,
                "bpmn:sequenceFlow",
                {
                    "id": f"Flow_{idx}_split_XOR",
                    "sourceRef": base_source_id,
                    "targetRef": source_id
                }
            )
        elif f"split_{idx}" in gateways:
            source_id = gateways[f"split_{idx}"].attrib['id']
            # Cria o fluxo da Activity -> Gateway split
            ET.SubElement(
                process,
                "bpmn:sequenceFlow",
                {
                    "id": f"Flow_{idx}_split",
                    "sourceRef": base_source_id,
                    "targetRef": source_id
                }
            )
        else:
            # Sem gateway de split, a própria atividade é o "source"
            source_id = base_source_id

        # --------------------------------------------------------------------
        # 3.2) Criar fluxos para cada "destino" cujo valor == 1
        # --------------------------------------------------------------------
        for target_idx, value in enumerate(row):
            if value == 1:
                # Decide se há gateway de join_XOR ou join normal ou nada
                if f"join_XOR_{target_idx}" in gateways:
                    xor_join_id = gateways[f"join_XOR_{target_idx}"].attrib["id"]

                    # Source -> XOR Join
                    ET.SubElement(
                        process,
                        "bpmn:sequenceFlow",
                        {
                            "id": f"Flow_{idx}_join_XOR_{target_idx}",
                            "sourceRef": source_id,
                            "targetRef": xor_join_id
                        }
                    )
                    # XOR Join -> Activity
                    ET.SubElement(
                        process,
                        "bpmn:sequenceFlow",
                        {
                            "id": f"Flow_join_XOR_{target_idx}",
                            "sourceRef": xor_join_id,
                            "targetRef": f"Activity_{target_idx}"
                        }
                    )
                elif f"join_{target_idx}" in gateways:
                    normal_join_id = gateways[f"join_{target_idx}"].attrib["id"]

                    # Source -> join normal
                    ET.SubElement(
                        process,
                        "bpmn:sequenceFlow",
                        {
                            "id": f"Flow_{idx}_{target_idx}",
                            "sourceRef": source_id,
                            "targetRef": normal_join_id
                        }
                    )
                    # join normal -> Activity
                    ET.SubElement(
                        process,
                        "bpmn:sequenceFlow",
                        {
                            "id": f"Flow_join_{target_idx}",
                            "sourceRef": normal_join_id,
                            "targetRef": f"Activity_{target_idx}"
                        }
                    )
                else:
                    # Fluxo direto
                    ET.SubElement(
                        process,
                        "bpmn:sequenceFlow",
                        {
                            "id": f"Flow_{idx}_{target_idx}",
                            "sourceRef": source_id,
                            "targetRef": f"Activity_{target_idx}"
                        }
                    )

    # --------------------------------------------------------
    # 4) Ajustar o fluxo inicial (tirar Activity_0 como fonte direta)
    # --------------------------------------------------------
    begin_activity_id = "Activity_0"
    for elem in list(process):
        if elem.tag.endswith("sequenceFlow") and elem.attrib.get("sourceRef") == begin_activity_id:
            # Cria um novo fluxo partindo do StartEvent_1 -> target
            new_flow = ET.SubElement(
                process,
                "bpmn:sequenceFlow",
                {
                    "id": f"Flow_Start_{elem.attrib['targetRef']}",
                    "sourceRef": "StartEvent_1",
                    "targetRef": elem.attrib["targetRef"]
                }
            )
            # Remove o fluxo antigo
            process.remove(elem)

    # --------------------------------------------------------
    # 5) Ajustar o fluxo final (ligar a última atividade ao EndEvent_1)
    # --------------------------------------------------------
    end_activity_id = f"Activity_{len(alphabet) - 1}"
    for elem in list(process):
        if elem.tag.endswith("sequenceFlow") and elem.attrib.get("targetRef") == end_activity_id:
            ET.SubElement(
                process,
                "bpmn:sequenceFlow",
                {
                    "id": f"Flow_{elem.attrib['sourceRef']}_End",
                    "sourceRef": elem.attrib["sourceRef"],
                    "targetRef": "EndEvent_1"
                }
            )
            process.remove(elem)

    # --------------------------------------------------------
    # 6) Se deseja remover Activity_0 e Activity_(fim) do diagrama:
    #    (depende do seu caso de uso e do que essas atividades significam)
    # --------------------------------------------------------
    for activity in list(process):
        if activity.tag.endswith("task") and activity.attrib.get("id") == "Activity_0":
            process.remove(activity)

    for activity in list(process):
        if activity.tag.endswith("task") and activity.attrib.get("id") == end_activity_id:
            process.remove(activity)

    # --------------------------------------------------------
    # 7) Retornar o XML resultante
    # --------------------------------------------------------
    return ET.tostring(bpmn, encoding="utf-8").decode("utf-8")

@decorators.measure_time
def create_pn(cromossome, alphabet, island, generation, input_log_name, round, cache_petri_net):
    petrinet, initial_marking, final_marking = create_petri_net(cromossome, alphabet, input_log_name, round, island, generation, cache_petri_net)
    input_log_name = input_log_name.replace("\\", "-").replace("/", "-").replace(":", "-").replace(" ", "-")
    pm4py.write_pnml(petrinet, initial_marking, final_marking, 'petri-nets/temp/' + str(input_log_name) + '-' + str(round) + '-' + str(island) + '-' + str(generation))
    return

@decorators.measure_time
def write_and_show_best_pn(best_island_number, input_log_name, round, generation_best, bestone_file, xes_log, fitness_weight, precision_weight, generalization_weight, simplicity_weight):
    petrinet_name = 'petri-nets/temp/' + str(input_log_name.replace("\\", "-").replace("/", "-")) + '-' + str(round) + '-' + str(best_island_number) + '-' + str(generation_best) + '.pnml'
    petrinet, initial_marking, final_marking = pm4py.read_pnml(petrinet_name)
    fitness = pm4py.fitness_alignments(xes_log, petrinet, initial_marking, final_marking)
    precision = pm4py.precision_alignments(xes_log, petrinet, initial_marking, final_marking)
    generaliz = pm4py.generalization_tbr(xes_log, petrinet, initial_marking, final_marking)
    #simplic = ft.structuredness(petrinet, initial_marking, final_marking)
    simplic = pm4py.simplicity_petri_net(petrinet, initial_marking, final_marking, variant='arc_degree')
    f_score = (fitness_weight + precision_weight + generalization_weight + simplicity_weight) / ((fitness_weight / fitness['log_fitness']) + (precision_weight / precision) + (generalization_weight / generaliz) + (simplicity_weight / simplic))
    print('HM: %.15f' % f_score, 'fitness: %.15f' % fitness['log_fitness'], 'precision: %.15f' % precision, 'generalization: %.15f' % generaliz, 'simplicity: %.15f' % simplic)
    bestone_file = bestone_file.replace("/", "-").replace(".", "-").replace("\\", "-")
    pm4py.write_pnml(petrinet, initial_marking, final_marking, 'petri-nets/best/' + bestone_file)
    pm4py.view_petri_net(petrinet, initial_marking, final_marking)
    bpmn_model = pm4py.convert_to_bpmn(petrinet, initial_marking, final_marking)
    pm4py.view_bpmn(bpmn_model)
    return

@decorators.measure_time
def create_petri_net(cromossome, alphabet, input_log_name, round, island, generation, cache_petri_net):
    cromossomo_hash = hash_cromossomo_petri_net(cromossome)
    raw = cache_petri_net.get(cromossomo_hash)
    if raw is not None:                                  # cache hit
        net, im, fm = cloudpickle.loads(raw)
        return net, im, fm
    bpmn_xml = generate_bpmn_from_cromossome(cromossome, alphabet)
    time = str(dt.datetime.now()).replace(" ", "-").replace(":", "-").replace(" ", "-").replace(".", "-")
    bpmn_xml_file_path = 'petri-nets/temp/' + str(input_log_name.replace("\\", "-").replace("/", "-").replace(":", "-").replace(" ", "-")) + '-' + str(round) + '-' + str(island) + '-' + str(generation) + '-' + str(time) + '.bpmn'
    with open(bpmn_xml_file_path, "w", encoding="utf-8") as file:
        file.write(bpmn_xml)
    bpmn_model = pm4py.read_bpmn(bpmn_xml_file_path)
    net, im, fm = pm4py.convert_to_petri_net(bpmn_model)
    net = pm4py.reduce_petri_net_invisibles(net)
    net, im, fm = pm4py.reduce_petri_net_implicit_places(net, im, fm)
    os.remove(bpmn_xml_file_path)
    cache_petri_net[cromossomo_hash] = cloudpickle.dumps((net, im, fm))
    return net, im, fm

@decorators.measure_time
def is_sound(cromossome, alphabet, input_log_name, round, island, generation, cache_soundness, cache_petri_net):
    cromossomo_hash = hash_cromossomo_soundness(cromossome)
    if cromossomo_hash in cache_soundness:
        if isinstance(cache_soundness.get(cromossomo_hash), bool):
            return cache_soundness[cromossomo_hash]
    petrinet, initial_marking, final_marking = create_petri_net(cromossome, alphabet, input_log_name, round, island, generation, cache_petri_net)
    res = pm4py.check_soundness(petrinet, initial_marking, final_marking)
    if isinstance(res[0], bool):
        if cromossomo_hash not in cache_soundness:
            cache_soundness[cromossomo_hash] = res[0]
    return res[0]

@decorators.measure_time
def hash_cromossomo_soundness(cromossomo):
    cromossomo_str = str(cromossomo.tolist())
    return hashlib.sha256(cromossomo_str.encode()).hexdigest()

@decorators.measure_time
def hash_cromossomo_petri_net(cromossomo):
    cromossomo_str = str(cromossomo.tolist())
    return hashlib.sha256(cromossomo_str.encode()).hexdigest()