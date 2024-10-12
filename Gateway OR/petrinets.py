import pm4py
import xml.etree.ElementTree as ET
import numpy as np
import datetime as dt
import os

def get_gateway_type(value):
    if value == 3:
        return "bpmn:inclusiveGateway"
    elif value == 2:
        return "bpmn:parallelGateway"
    elif value == 1:
        return "bpmn:exclusiveGateway"

def generate_bpmn_from_cromossome(cromossome, alphabet):
    bpmn = ET.Element("bpmn:definitions", {"xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance", "xmlns:bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL", "xmlns:bpmndi": "http://www.omg.org/spec/BPMN/20100524/DI", "xmlns:dc": "http://www.omg.org/spec/DD/20100524/DC", "xmlns:di": "http://www.omg.org/spec/DD/20100524/DI", "id": "Definitions_001", "targetNamespace": "http://bpmn.io/schema/bpmn"})
    process = ET.SubElement(bpmn, "bpmn:process", {"id": "Process_001", "isExecutable": "false"})
    ET.SubElement(process, "bpmn:startEvent", {"id": "StartEvent_1"})
    ET.SubElement(process, "bpmn:endEvent", {"id": "EndEvent_1"})
    activities = []
    gateways = {}
    for idx, activity in enumerate(alphabet):
        task = ET.SubElement(process, "bpmn:task", {"id": f"Activity_{idx}", "name": activity})
        activities.append(task)
    for idx, value in enumerate(cromossome[:, -1]):
        if value > 0:
            gateway_type = get_gateway_type(value)
            gateway = ET.SubElement(process, gateway_type, {"id": f"Gateway_split_{idx}"})
            gateways[f"split_{idx}"] = gateway
    for idx, value in enumerate(cromossome[-1, :-1]):
        if value > 0:
            gateway_type = get_gateway_type(value)
            gateway = ET.SubElement(process, gateway_type, {"id": f"Gateway_join_{idx}"})
            gateways[f"join_{idx}"] = gateway
    for idx, row in enumerate(cromossome[:-1]):
        if np.sum(row[:-1]) > 1 and row[-1] == 0:
            gateway = ET.SubElement(process, "bpmn:exclusiveGateway", {"id": f"Gateway_split_XOR_{idx}"})
            gateways[f"split_XOR_{idx}"] = gateway
    for idx in range(cromossome.shape[1] - 1):
        column = cromossome[:-1, idx]
        if np.sum(column) > 1 and cromossome[-1, idx] == 0:
            gateway = ET.SubElement(process, "bpmn:exclusiveGateway", {"id": f"Gateway_join_XOR_{idx}"})
            gateways[f"join_XOR_{idx}"] = gateway
    for idx, row in enumerate(cromossome):
        source_id = f"Activity_{idx}"
        if f"split_{idx}" in gateways:
            source_id = gateways[f"split_{idx}"].attrib['id']
            flow = ET.SubElement(process, "bpmn:sequenceFlow", {"id": f"Flow_{idx}_split", "sourceRef": f"Activity_{idx}", "targetRef": source_id})
        if f"split_XOR_{idx}" in gateways:
            source_id = gateways[f"split_XOR_{idx}"].attrib['id']
            flow = ET.SubElement(process, "bpmn:sequenceFlow", {"id": f"Flow_{idx}_split_XOR", "sourceRef": f"Activity_{idx}", "targetRef": source_id})
        for target_idx, value in enumerate(row):
            if value == 1:
                target_id = f"Activity_{target_idx}"
                if f"join_{target_idx}" in gateways:
                    target_id = gateways[f"join_{target_idx}"].attrib['id']
                flow = ET.SubElement(process, "bpmn:sequenceFlow", {"id": f"Flow_{idx}_{target_idx}", "sourceRef": source_id, "targetRef": target_id})
                if f"join_{target_idx}" in gateways:
                    flow = ET.SubElement(process, "bpmn:sequenceFlow", {"id": f"Flow_join_{target_idx}", "sourceRef": gateways[f"join_{target_idx}"].attrib['id'], "targetRef": f"Activity_{target_idx}"})
                if f"join_XOR_{target_idx}" in gateways:
                    target_id = gateways[f"join_XOR_{target_idx}"].attrib['id']
                    flow = ET.SubElement(process, "bpmn:sequenceFlow", {"id": f"Flow_{idx}_join_XOR", "sourceRef": source_id, "targetRef": target_id})
                    flow = ET.SubElement(process, "bpmn:sequenceFlow", {"id": f"Flow_join_XOR_{target_idx}", "sourceRef": target_id, "targetRef": f"Activity_{target_idx}"})
    begin_activity_id = "Activity_0"
    for flow in process:
        if flow.tag.endswith("sequenceFlow") and flow.attrib.get("sourceRef") == begin_activity_id:
            new_flow = ET.SubElement(process, "bpmn:sequenceFlow", {"id": f"Flow_Start_{flow.attrib['targetRef']}", "sourceRef": "StartEvent_1", "targetRef": flow.attrib["targetRef"]})
            process.remove(flow)
    end_activity_id = f"Activity_{len(alphabet) - 1}"
    for flow in process:
        if flow.tag.endswith("sequenceFlow") and flow.attrib.get("targetRef") == end_activity_id:
            ET.SubElement(process, "bpmn:sequenceFlow", {"id": f"Flow_{flow.attrib['sourceRef']}_End", "sourceRef": flow.attrib["sourceRef"], "targetRef": "EndEvent_1"})
            process.remove(flow)
    for activity in process:
        if activity.tag.endswith("task") and activity.attrib.get("id") == "Activity_0":
            process.remove(activity)
    end_activity_id = f"Activity_{len(alphabet) - 1}"
    for activity in process:
        if activity.tag.endswith("task") and activity.attrib.get("id") == end_activity_id:
            process.remove(activity)
    return ET.tostring(bpmn, encoding="utf-8").decode("utf-8")

def create_pn(cromossome, alphabet, island, generation, log_name, round):
    petrinet, initial_marking, final_marking = create_petri_net(cromossome, alphabet, log_name, round, island, generation)
    log_name = log_name.replace("\\", "-").replace("/", "-")
    pm4py.write_pnml(petrinet, initial_marking, final_marking, 'petri-nets/temp/' + str(log_name) + '-' + str(round) + '-' + str(island) + '-' + str(generation))
    return

def write_and_show_best_pn(best_island_number, log_name, round, generation_best, bestone_file):
    petrinet_name = 'petri-nets/temp/' + str(log_name.replace("\\", "-").replace("/", "-")) + '-' + str(round) + '-' + str(best_island_number) + '-' + str(generation_best) + '.pnml'
    petrinet, initial_marking, final_marking = pm4py.read_pnml(petrinet_name)
    bestone_file = bestone_file.replace("/", "-").replace(".", "-").replace("\\", "-")
    pm4py.write_pnml(petrinet, initial_marking, final_marking, 'petri-nets/best/' + bestone_file)
    pm4py.view_petri_net(petrinet, initial_marking, final_marking)
    bpmn_model = pm4py.convert_to_bpmn(petrinet, initial_marking, final_marking)
    pm4py.view_bpmn(bpmn_model)
    return

def create_petri_net(cromossome, alphabet, log_name, round, island, generation):
    bpmn_xml = generate_bpmn_from_cromossome(cromossome, alphabet)
    time = str(dt.datetime.now()).replace(" ", "-").replace(":", "-").replace(".", "-")
    file_path = 'petri-nets/temp/' + str(log_name.replace("\\", "-").replace("/", "-")) + '-' + str(round) + '-' + str(island) + '-' + str(generation) + '-' + str(time) + '.bpmn'
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(bpmn_xml)
    bpmn_model = pm4py.read_bpmn(file_path)
    net, im, fm = pm4py.convert_to_petri_net(bpmn_model)
    net = pm4py.reduce_petri_net_invisibles(net)
    net, im, fm = pm4py.reduce_petri_net_implicit_places(net, im, fm)
    os.remove(file_path)
    return net, im, fm

def is_sound(cromossome, alphabet, log_name, round, island, generation):
    petrinet, initial_marking, final_marking = create_petri_net(cromossome, alphabet, log_name, round, island, generation)
    res = pm4py.check_soundness(petrinet, initial_marking, final_marking)
    return res[0]