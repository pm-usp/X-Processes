from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from pm4py.objects.bpmn.obj import BPMN
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import reachability_graph, final_marking
from pm4py.objects.petri_net.utils import reduction as pn_reduction
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union, Literal, Generator, Sequence
import heapq
import math
import networkx as nx
import pandas as pd


_NODE_TASK = "TASK"
_NODE_AND = "AND"
_NODE_XOR = "XOR"
_NODE_OR = "OR"

#
# Auxiliary functions for BPMN
#

#
# Get Outgoing Flows (BPMN Node)
#
def get_outgoing_flows_bpmn(bpmn: BPMN, node_id: str):
    return [f for f in bpmn.get_flows() if f.source.id == node_id]
#enddef

#
# Get Incoming Flows (BPMN Node)
#
def get_incoming_flows_bpmn(bpmn: BPMN, node_id: str):
    return [f for f in bpmn.get_flows() if f.target.id == node_id]
#enddef

#
# Cross Connectivity Metric
# This is the recursive DFS function to find all paths
# AI-Generated (Gemini 2.5 Pro)
#
def _find_all_paths_generator(
    graph: Dict[str, List[str]],
    start_node: str,
    end_node: str,
    path: List[str] = None
) -> Generator[List[str], None, None]:
    """
    Finds all acyclic paths between a start and end node using a memory-efficient
    generator with a corrected pathing mechanism.
    """
    # 1. Correctly initialize the path on the first call
    if path is None:
        path = []
    #endif

    # 2. Use append/pop on the SAME list to avoid creating copies
    path.append(start_node)

    if start_node == end_node:
        # 3. Yield a COPY of the found path
        yield list(path)
    else:
        if start_node in graph:
            for neighbor in graph.get(start_node, []):
                if neighbor not in path:  # Avoid cycles
                    # 4. Recursively yield from the deeper calls
                    yield from _find_all_paths_generator(graph, neighbor, end_node, path)
                #endif
            #endfor
        #endif
    #endif

    # 5. Backtrack: remove the current node from the path to explore other branches
    path.pop()
#enddef

def _find_all_paths(graph: Dict[str, List[str]], start_node: str, end_node: str, path: List[str]) -> List[List[str]]:
    """
    Finds all acyclic paths between a start and end node using DFS.
    """
    path = path + [start_node]
    if start_node == end_node: return [path]
    if start_node not in graph: return []

    paths = []
    for neighbor in graph[start_node]:
        if neighbor not in path:  # Avoid cycles
            new_paths = _find_all_paths(graph, neighbor, end_node, path)
            for new_path in new_paths:
                paths.append(new_path)
            #endfor
        #endif
    #endfor

    return paths
#enddef

#
# AI-Generated (Gemini 2.5 Pro): Auxiliar function for Advanced Structural Appropriateness
# Find Redundant Invisible Tasks
#
# def find_redundant_invisible_tasks(net: PetriNet):
#     """
#     Finds redundant invisible transitions (tir).
#
#     Based on Rozinat (2008), a redundant invisible task is one that
#     can be removed without changing model behavior. The most common
#     structural example is a silent transition (label=None) that forms a
#     simple, non-choice sequence (p_in -> t_silent -> p_out).
#
#     This transition is redundant if it's the *only* thing leaving p_in
#     and the *only* thing entering p_out.
#     """
#     redundant_tasks = set()
#
#     invisible_transitions = {t for t in net.transitions if t.label is None}
#
#     for t in invisible_transitions:
#         # A simple redundant task has exactly one input and one output place
#         if len(t.in_arcs) == 1 and len(t.out_arcs) == 1:
#             # Get the single input and output place
#             # Note: In pm4py, arc.source and arc.target are Place/Transition objects
#             try:
#                 p_in = list(t.in_arcs)[0].source
#                 p_out = list(t.out_arcs)[0].target
#
#             except (IndexError, AttributeError):
#                 continue  # Should not happen with valid PetriNet object
#
#             # Check if 't' is the *only* output of its input place
#             # and the *only* input of its output place.
#             if len(p_in.out_arcs) == 1 and len(p_out.in_arcs) == 1:
#                 # This transition is structurally redundant
#                 redundant_tasks.add(t)
#
#     return redundant_tasks
# #enddef

#
# AI-Generated (Gemini 2.5 Pro): Auxiliar function for Advanced Structural Appropriateness
# Find Alternative Duplicate Tasks
#
# def find_alternative_duplicate_tasks(net:PetriNet, initial_marking:Marking, final_marking:Marking = None, num_traces=10000):
#     """
#     Finds alternative duplicate tasks (tda).
#
#     Based on Rozinat (2008), these are duplicate-labeled transitions
#     that *never* occur together in the same execution (i.e., they are
#     in an exclusive choice).
#
#     This is approximated using simulation (playout), which is a practical
#     proxy for the full "state space analysis" mentioned in the paper.
#     """
#
#     # 1. Find all groups of duplicate-labeled transitions
#     dupe_groups = defaultdict(set)
#     for t in net.transitions:
#         if t.label is not None:
#             dupe_groups[t.label].add(t)
#
#     # Filter to only groups that actually have duplicates
#     dupe_groups = {label: transitions for label, transitions in dupe_groups.items() if len(transitions) > 1}
#
#     if not dupe_groups:
#         return set()  # No duplicates, so no alternative duplicates
#
#     # 2. Run playout to simulate the state space
#     try:
#         # PM4Py playout returns an EventLog object (a list of traces)
#         log = simulator.apply(
#             net=net,
#             initial_marking=initial_marking,
#             final_marking=final_marking,
#             parameters={
#                 "no_traces": num_traces,
#                 simulator.Parameters.ACTIVITY_KEY: lambda t: t
#             },
#             variant=simulator.Variants.BASIC_PLAYOUT
#         )
#
#     except Exception as e:
#         print(f"Warning: Playout simulation failed. Cannot calculate TDA. Error: {e}")
#         return set()
#
#     # 3. Find all pairs of duplicates that *do* co-occur
#     co_occurring_pairs = set()
#
#     for trace in log:
#         # Get the set of unique transitions that fired in this trace
#         # A trace is a list of events (dicts)
#         fired_transitions_in_trace = set()
#         for event in trace:
#             # The activity_key lambda stores the Transition object here
#             activity = event.get("concept:name")
#             if isinstance(activity, PetriNet.Transition):
#                 fired_transitions_in_trace.add(activity)
#
#         for label, group in dupe_groups.items():
#             # Find which transitions from this group fired
#             fired_dupes = fired_transitions_in_trace.intersection(group)
#
#             # If 2 or more fired, add all pairs to the co-occurring set
#             if len(fired_dupes) > 1:
#                 group_list = list(fired_dupes)
#                 for i in range(len(group_list)):
#                     for j in range(i + 1, len(group_list)):
#                         # Create a stable, hashable key for the pair
#                         pair = tuple(sorted(group_list, key=lambda t: t.name))
#                         co_occurring_pairs.add(pair)
#
#     # 4. Find all alternative transitions
#     alternative_transitions = set()
#
#     for label, group in dupe_groups.items():
#         group_list = list(group)
#         # Check every possible pair within the group
#         for i in range(len(group_list)):
#             for j in range(i + 1, len(group_list)):
#                 t1, t2 = group_list[i], group_list[j]
#                 pair = tuple(sorted((t1, t2), key=lambda t: t.name))
#
#                 # If this pair *never* co-occurred, they are alternative
#                 if pair not in co_occurring_pairs:
#                     alternative_transitions.add(t1)
#                     alternative_transitions.add(t2)
#
#     return alternative_transitions
# #enddef


def _all_simple_paths(
    graph: Dict[str, List[str]],
    start: str,
    end: str,
) -> Iterable[List[str]]:
    if start not in graph or end not in graph:
        return

    stack: List[Tuple[str, List[str]]] = [(start, [start])]

    while stack:
        node, path = stack.pop()
        if node == end:
            yield path
            continue

        for succ in graph.get(node, []):
            if succ not in path:
                stack.append((succ, path + [succ]))

#
# Reduce Trivial Constructs (Auxiliar)
# Auxiliar function for Structuredness Ratio
#
def reduce_trivial_constructs_aux(bpmn: BPMN):
    changed = False

    # Step 1: Identify preserved activities
    preserved_activities = set()
    for node in bpmn.get_nodes():
        if isinstance(node, BPMN.StartEvent):
            for f in get_outgoing_flows_bpmn(bpmn, node.id):
                if isinstance(f.target, BPMN.Activity):
                    preserved_activities.add(f.target.id)
                #endif
            #endfor
        elif isinstance(node, BPMN.EndEvent):
            for f in get_incoming_flows_bpmn(bpmn, node.id):
                if isinstance(f.source, BPMN.Activity):
                    preserved_activities.add(f.source.id)
                #endif
            #endfor
        #endif
    #endfor

    # Step 2: Traverse all nodes and remove trivial constructs
    nodes = list(bpmn.get_nodes())
    for node in nodes:
        # Mark flag as False
        to_delete = False

        # Get Inbound and Outbound arcs
        in_flows = get_incoming_flows_bpmn(bpmn, node.id)
        out_flows = get_outgoing_flows_bpmn(bpmn, node.id)

        # Do not consider Activities after Start and before End
        if node.id in preserved_activities:
            continue
        #endif

        # Only consider nodes with 1 incoming and 1 outgoing flow (trivial pass-through)
        if len(in_flows) != 1 or len(out_flows) != 1:
            continue
        #endif

        prev_node = in_flows[0].source
        next_node = out_flows[0].target

        # Delete: Alone Activity between two Gateways
        if isinstance(prev_node, BPMN.Gateway) and isinstance(next_node, BPMN.Gateway):
            to_delete = True
        #endif

        # Delete: Activity in between two activities
        if isinstance(prev_node, BPMN.Activity) and isinstance(next_node, BPMN.Activity):
            to_delete = True
        #endif

        # Delete: Activity before and after Gateway with activity in between
        if ((isinstance(prev_node, BPMN.Gateway) and isinstance(next_node, BPMN.Activity)) or
                (isinstance(prev_node, BPMN.Activity) and isinstance(next_node, BPMN.Gateway))):
            to_delete = True
        #endif

        if to_delete:
            # Remove current node and its flows
            bpmn.remove_flow(in_flows[0])
            bpmn.remove_flow(out_flows[0])
            bpmn.remove_node(node)

            # Add new direct flow
            bpmn.add_flow(BPMN.Flow(prev_node, next_node))

            #print(f"Eliminated: {node.id}")
            changed = True
        #endif
    #endfor

    return changed
#enddef

#
# Reduce Trivial Constructs
#
def reduce_trivial_constructs(bpmn: BPMN):
    while reduce_trivial_constructs_aux(bpmn):
        print("Reduce Trivial Constructs: completed")
        pass
    return bpmn
#enddef


def _count_redundant_invisible_tasks_strict(
    net: PetriNet,
    im: Marking,
    fm: Optional[Marking] = None,
) -> int:
    """
    Counts |T_IR| using Murata-style structural reduction rules.

    Assumption (requested by user):
    - The reduction used here correctly removes/merges only redundant invisible
      transitions without changing the behavior.
    """
    if fm is None:
        try:
            fm = final_marking.discover_final_marking(net)
        except Exception:
            fm = Marking()

    net_copy, im_copy, fm_copy = deepcopy((net, im, fm))

    invisibles_before = [t for t in net_copy.transitions if t.label is None]
    n_before = len(invisibles_before)

    net_copy, im_copy, fm_copy = pn_reduction.apply_reset_inhibitor_net_reduction(
        net_copy, im_copy, fm_copy
    )

    invisibles_after = [t for t in net_copy.transitions if t.label is None]
    n_after = len(invisibles_after)

    tir = n_before - n_after
    return max(0, tir)


def _find_alternative_duplicate_tasks_marking_flow(
    net: PetriNet,
    im: Marking,
) -> Set[PetriNet.Transition]:
    """
    Computes T_DA, the set of alternative duplicate tasks, from the marking flow.

    Definition (Rozinat 2008, Metric 7):
    - Duplicate tasks (same label) that never happen together in one
      execution sequence should be considered alternative duplicates.

    Implementation choices:
    - Uses the marking-flow-based reachability graph of pm4py, which
      works on the actual Transition objects (no dependence on .name).
    - Assumes a sound WF-net, so every reachable marking lies on a
      complete run from im to some final marking.
    - For each label group with >= 2 transitions, we mark a transition as
      "alternative duplicate" iff it does not co-occur with any other
      duplicate transition of the same label in the marking graph.
      (This is the less punitive rule for 3+ duplicates.)
    """
    # Group visible transitions by label
    dupe_groups: Dict[str, List[PetriNet.Transition]] = defaultdict(list)
    for t in net.transitions:
        if t.label is not None:
            dupe_groups[t.label].append(t)

    dupe_groups = {lbl: group for lbl, group in dupe_groups.items() if len(group) > 1}
    if not dupe_groups:
        return set()

    # Build marking flow: incoming/outgoing transitions over reachable markings
    incoming_transitions, outgoing_transitions, _ = reachability_graph.marking_flow_petri(
        net, im, return_eventually_enabled=True
    )
    # incoming_transitions is unused, but kept for parity with the call contract
    _ = incoming_transitions

    # Build graph of markings and mapping transition -> edges
    G = nx.DiGraph()
    trans_to_edges: Dict[PetriNet.Transition, List[Tuple[object, object]]] = defaultdict(list)

    for m, succ in outgoing_transitions.items():
        for t, m_next in succ.items():
            G.add_edge(m, m_next)
            trans_to_edges[t].append((m, m_next))

    alternative_transitions: Set[PetriNet.Transition] = set()

    for label, group in dupe_groups.items():
        group_list = list(group)

        # Liveness filter: consider only transitions that actually fire in the graph
        active = [t for t in group_list if bool(trans_to_edges.get(t))]
        if len(active) < 2:
            continue

        # Build undirected "co-occurrence" relation inside this label group
        cooccurs_with: Dict[PetriNet.Transition, Set[PetriNet.Transition]] = {
            t: set() for t in active
        }

        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                t1 = active[i]
                t2 = active[j]
                if _pair_cooccurs_in_marking_graph(G, t1, t2, trans_to_edges):
                    cooccurs_with[t1].add(t2)
                    cooccurs_with[t2].add(t1)

        # Less punitive rule for 3+ duplicates:
        # penalize a transition only if it does not co-occur with any other duplicate
        for t in active:
            if len(cooccurs_with[t]) == 0:
                alternative_transitions.add(t)

    return alternative_transitions


def _pair_cooccurs_in_marking_graph(
    G: nx.DiGraph,
    t1: PetriNet.Transition,
    t2: PetriNet.Transition,
    trans_to_edges: Dict[PetriNet.Transition, List[Tuple[object, object]]],
) -> bool:
    """
    Returns True if there exists some path in the marking graph where
    both t1 and t2 fire (in any order).

    Under the sound WF-net assumption, any reachable marking lies on a
    complete run, so "path in the marking graph" is equivalent to
    "execution sequence" in the sense of Rozinat (2008).
    """
    edges_t1 = trans_to_edges.get(t1, [])
    edges_t2 = trans_to_edges.get(t2, [])

    if not edges_t1 or not edges_t2:
        return False

    sources_t1 = {u for (u, v) in edges_t1}
    targets_t1 = {v for (u, v) in edges_t1}
    sources_t2 = {u for (u, v) in edges_t2}
    targets_t2 = {v for (u, v) in edges_t2}

    # Case A: t1 before t2
    for start in targets_t1:
        reachable = nx.descendants(G, start)
        if start in sources_t2 or not sources_t2.isdisjoint(reachable):
            return True

    # Case B: t2 before t1
    for start in targets_t2:
        reachable = nx.descendants(G, start)
        if start in sources_t1 or not sources_t1.isdisjoint(reachable):
            return True

    return False









def _build_augmented_graph(bpmn: BPMN) -> Tuple[Set[str], List[Tuple[str, str]], Dict[str, str], Dict[str, str]]:
    include_events = True
    with_labels = False
    allowed = (BPMN.Task, BPMN.ParallelGateway, BPMN.ExclusiveGateway, BPMN.InclusiveGateway)
    if include_events:
        allowed = allowed + (BPMN.StartEvent, BPMN.EndEvent)

    base_nodes = {n.id: n for n in bpmn.get_nodes() if isinstance(n, allowed)}

    base_edges: List[Tuple[str, str]] = []
    for f in bpmn.get_flows():
        if not isinstance(f, BPMN.SequenceFlow):
            continue
        u = f.source.id
        v = f.target.id
        if u in base_nodes and v in base_nodes:
            base_edges.append((u, v))

    kind: Dict[str, str] = {}
    label: Dict[str, str] = {}

    for nid, n in base_nodes.items():
        if isinstance(n, (BPMN.Task, BPMN.StartEvent, BPMN.EndEvent)):
            kind[nid] = _NODE_TASK
        elif isinstance(n, BPMN.ParallelGateway):
            kind[nid] = _NODE_AND
        elif isinstance(n, BPMN.ExclusiveGateway):
            kind[nid] = _NODE_XOR
        elif isinstance(n, BPMN.InclusiveGateway):
            kind[nid] = _NODE_OR
        else:
            kind[nid] = _NODE_TASK

        if with_labels:
            base_name = getattr(n, "name", None)
            base_name = base_name.strip() if isinstance(base_name, str) else ""
            if not base_name:
                base_name = type(n).__name__
            base_name = base_name.replace("\n", " ").replace("\r", " ").strip()
            label[nid] = f"{base_name} ({nid})"

    in_map: Dict[str, List[str]] = {nid: [] for nid in base_nodes}
    out_map: Dict[str, List[str]] = {nid: [] for nid in base_nodes}
    for u, v in base_edges:
        out_map[u].append(v)
        in_map[v].append(u)

    def is_gateway_id(nid: str) -> bool:
        return kind[nid] in (_NODE_AND, _NODE_XOR, _NODE_OR)

    has_join = {nid: (not is_gateway_id(nid) and len(in_map[nid]) >= 2) for nid in base_nodes}
    has_split = {nid: (not is_gateway_id(nid) and len(out_map[nid]) >= 2) for nid in base_nodes}

    nodes: Set[str] = set(base_nodes.keys())

    for nid in base_nodes:
        if has_join[nid]:
            vj = f"VJ_{nid}"
            nodes.add(vj)
            kind[vj] = _NODE_XOR
            if with_labels:
                label[vj] = f"Implicit XOR-join ({nid})"
        if has_split[nid]:
            vs = f"VS_{nid}"
            nodes.add(vs)
            kind[vs] = _NODE_AND
            if with_labels:
                label[vs] = f"Implicit AND-split ({nid})"

    edges_mult: List[Tuple[str, str]] = []

    for nid in base_nodes:
        if has_join[nid]:
            edges_mult.append((f"VJ_{nid}", nid))
        if has_split[nid]:
            edges_mult.append((nid, f"VS_{nid}"))

    for u, v in base_edges:
        u2 = f"VS_{u}" if has_split[u] else u
        v2 = f"VJ_{v}" if has_join[v] else v
        edges_mult.append((u2, v2))

    return nodes, edges_mult, kind, label


def _is_virtual(nid: str) -> bool:
    return nid.startswith("VJ_") or nid.startswith("VS_")


def _compute_degrees(
    nodes: Set[str],
    edges: List[Tuple[str, str]],
    *,
    deduplicate: bool,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    indeg: Dict[str, int] = {nid: 0 for nid in nodes}
    outdeg: Dict[str, int] = {nid: 0 for nid in nodes}

    if deduplicate:
        seen: Set[Tuple[str, str]] = set()
        for u, v in edges:
            if (u, v) in seen:
                continue
            seen.add((u, v))
            outdeg[u] += 1
            indeg[v] += 1
    else:
        for u, v in edges:
            outdeg[u] += 1
            indeg[v] += 1

    return indeg, outdeg


def _compute_node_weights(
    nodes: Set[str],
    indeg: Dict[str, int],
    outdeg: Dict[str, int],
    kind: Dict[str, str],
) -> Dict[str, float]:
    """
    Pesos de no conforme Definition 1.
    OR calculado com inteiros exatos e Fraction, depois convertido para float.
    """
    w: Dict[str, float] = {}
    for nid in nodes:
        d = indeg[nid] + outdeg[nid]
        k = kind[nid]

        if k in (_NODE_TASK, _NODE_AND):
            w[nid] = 1.0
        elif k == _NODE_XOR:
            w[nid] = (1.0 / float(d)) if d > 0 else 1.0
        elif k == _NODE_OR:
            if d > 0:
                two_pow_d = 1 << d
                denom = two_pow_d - 1
                w_frac = Fraction(1, denom) + Fraction(two_pow_d - 2, denom) * Fraction(1, d)
                w[nid] = float(w_frac)
            else:
                w[nid] = 1.0
        else:
            raise ValueError(f"Unknown node kind: {k}")

        if not (0.0 < w[nid] <= 1.0):
            raise ValueError(f"Invalid weight for node {nid}: w={w[nid]}, kind={k}, d={d}")

    return w


def _build_adjacency_and_costs(
    nodes: Set[str],
    edges: List[Tuple[str, str]],
    w: Dict[str, float],
) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], float]]:
    """
    Usa arestas unicas (u,v) para Dijkstra, mas aceita lista com multiplicidade.
    """
    adjacency: Dict[str, Set[str]] = {nid: set() for nid in nodes}
    arc_cost: Dict[Tuple[str, str], float] = {}

    for u, v in edges:
        if v not in adjacency[u]:
            adjacency[u].add(v)
            prod = w[u] * w[v]
            if prod > 0.0:
                arc_cost[(u, v)] = -math.log(prod)

    return adjacency, arc_cost


def _dijkstra_shortest_paths(
    src: str,
    adjacency: Dict[str, Set[str]],
    arc_cost: Dict[Tuple[str, str], float],
    *,
    targets: Optional[Set[str]] = None,
) -> Dict[str, float]:
    dist: Dict[str, float] = {src: 0.0}
    visited: Set[str] = set()
    heap: List[Tuple[float, str]] = [(0.0, src)]

    remaining = None
    if targets is not None:
        remaining = set(targets)
        remaining.discard(src)

    while heap:
        du, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)

        if remaining is not None and u in remaining:
            remaining.remove(u)
            if not remaining:
                break

        for v in adjacency.get(u, set()):
            c = arc_cost.get((u, v))
            if c is None:
                continue
            nd = du + c
            od = dist.get(v)
            if od is None or nd < od:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    return dist


def calculate_connectivity_matrix(bpmn: BPMN) -> pd.DataFrame:
    """
    Matriz dirigida V(n1,n2), coerente com cross_connectivity.
    """
    include_events = True
    count_virtual_nodes = True
    count_parallel_flows_in_degree = True
    nodes, edges_mult, kind, label = _build_augmented_graph(bpmn)
    if not nodes:
        return pd.DataFrame()

    if count_virtual_nodes:
        eval_nodes = set(nodes)
    else:
        eval_nodes = {nid for nid in nodes if not _is_virtual(nid)}

    if len(eval_nodes) < 1:
        return pd.DataFrame()

    indeg, outdeg = _compute_degrees(nodes, edges_mult, deduplicate=not count_parallel_flows_in_degree)
    w = _compute_node_weights(nodes, indeg, outdeg, kind)

    adjacency, arc_cost = _build_adjacency_and_costs(nodes, edges_mult, w)

    ordered_ids = sorted(eval_nodes, key=lambda nid: label.get(nid, nid))
    ordered_labels = [label.get(nid, nid) for nid in ordered_ids]
    targets = set(ordered_ids)

    mat = pd.DataFrame(0.0, index=ordered_labels, columns=ordered_labels)

    for src_id in ordered_ids:
        src_lab = label.get(src_id, src_id)
        dist = _dijkstra_shortest_paths(src_id, adjacency, arc_cost, targets=targets)
        for tgt_id in ordered_ids:
            if tgt_id == src_id:
                continue
            tgt_lab = label.get(tgt_id, tgt_id)
            d_ = dist.get(tgt_id)
            mat.loc[src_lab, tgt_lab] = 0.0 if d_ is None else math.exp(-d_)

    mat["total"] = mat.sum(axis=1)
    return mat

@dataclass(frozen=True)
class EventClassifier:
    keys: Sequence[str] = ("concept:name",)
    sep: str = "||"

    def encode_event(self, e: dict) -> Optional[str]:
        parts = []
        for k in self.keys:
            v = e.get(k)
            if v is None:
                return None
            parts.append(str(v))
        return self.sep.join(parts)

    def encode_row(self, row: pd.Series) -> Optional[str]:
        parts = []
        for k in self.keys:
            if k not in row or pd.isna(row[k]):
                return None
            parts.append(str(row[k]))
        return self.sep.join(parts)


def _event_classes_from_log(
    log: Union[pd.DataFrame, EventLog],
    classifier: EventClassifier,
) -> Set[str]:
    classes: Set[str] = set()

    if isinstance(log, pd.DataFrame):
        missing_cols = [k for k in classifier.keys if k not in log.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame log: {missing_cols}")
        for _, row in log.iterrows():
            c = classifier.encode_row(row)
            if c is not None:
                classes.add(c)
        return classes

    for trace in log:
        for event in trace:
            c = classifier.encode_event(event)
            if c is not None:
                classes.add(c)
    return classes


def _is_invisible_transition_label(label: Optional[str]) -> bool:
    if label is None:
        return True
    s = str(label).strip()
    if s == "":
        return True
    return s.lower() in {"tau", "silent", "τ"}


# ==============================
# Errors
# ==============================

class ReachabilityError(RuntimeError):
    pass


class StateSpaceLimitError(ReachabilityError):
    pass


class TokenLimitError(ReachabilityError):
    pass


class DepthLimitError(ReachabilityError):
    pass


# ==============================
# Config
# ==============================

@dataclass(frozen=True)
class ReachabilityConfig:
    max_states: int = 200_000
    max_total_tokens: int = 2_000
    max_tokens_per_place: int = 200
    max_depth: Optional[int] = 200_000  # None desliga limite de profundidade


@dataclass(frozen=True)
class QWeights:
    w_op: float = 1.0
    w_sl: float = 5.0
    w_par: float = 0.5


# ==============================
# Basic helpers
# ==============================

def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _is_visible(t: Any) -> bool:
    return getattr(t, "label", None) is not None


def _visible_transitions(net: PetriNet) -> List[Any]:
    return [t for t in net.transitions if _is_visible(t)]


def _tokens_exceed_limits(m: Marking, cfg: ReachabilityConfig) -> bool:
    total = 0
    for _, v in m.items():
        if v is None:
            continue
        iv = int(v)
        if iv > cfg.max_tokens_per_place:
            return True
        if iv > 0:
            total += iv
            if total > cfg.max_total_tokens:
                return True
    return False


def _marking_key(m: Marking) -> frozenset:
    # Representação hashável, sem depender de Marking ser hashável no seu ambiente.
    # Ignora tokens 0 e None (equivalentes a ausência).
    items = []
    for p, v in m.items():
        if v is None:
            continue
        iv = int(v)
        if iv > 0:
            items.append((p, iv))
    return frozenset(items)


def _t_key(t: Any) -> str:
    name = getattr(t, "name", "")
    label = getattr(t, "label", None)
    if label is None:
        return f"{name}(tau)"
    return f"{name}('{label}')"


# ==============================
# PM4Py semantics wrappers
# ==============================

def _get_semantics():
    try:
        from pm4py.objects.petri_net.semantics import ClassicSemantics  # type: ignore
        return ClassicSemantics()
    except Exception:
        from pm4py.objects.petri_net import semantics as sem_mod  # type: ignore
        return sem_mod.ClassicSemantics()


def _enabled_transitions(sem: Any, net: PetriNet, m: Marking) -> List[Any]:
    return list(sem.enabled_transitions(net, m))


def _execute(sem: Any, t: Any, net: PetriNet, m: Marking) -> Optional[Marking]:
    return sem.execute(t, net, m)


# ==============================
# Reachability graph (materialize once, no silent truncation)
# ==============================

@dataclass(frozen=True)
class ReachabilityGraph:
    im_key: frozenset
    fm_key: frozenset
    adj: Dict[frozenset, List[Tuple[Any, frozenset]]]
    pred: Dict[frozenset, List[frozenset]]
    n_states: int


def _build_reachability_graph(
    net: PetriNet,
    im: Marking,
    fm: Marking,
    cfg: ReachabilityConfig
) -> ReachabilityGraph:
    """
    Materializa o grafo de alcançabilidade (até os limites).
    Sem falha silenciosa: se max_depth cortar qualquer expansão, levanta DepthLimitError.
    """
    if _tokens_exceed_limits(im, cfg):
        raise TokenLimitError("Initial marking excede limites de tokens, possível não-limitabilidade.")

    sem = _get_semantics()

    imk = _marking_key(im)
    fmk = _marking_key(fm)

    visited: Set[frozenset] = {imk}
    adj: Dict[frozenset, List[Tuple[Any, frozenset]]] = {imk: []}
    pred: Dict[frozenset, List[frozenset]] = {imk: []}

    q = deque([(im, imk, 0)])
    depth_cut = False

    while q:
        m_obj, mk, depth = q.popleft()

        if cfg.max_depth is not None and depth >= cfg.max_depth:
            depth_cut = True
            adj.setdefault(mk, [])
            pred.setdefault(mk, [])
            continue

        outs: List[Tuple[Any, frozenset]] = []

        for t in _enabled_transitions(sem, net, m_obj):
            m2 = _execute(sem, t, net, m_obj)
            if m2 is None:
                continue

            if _tokens_exceed_limits(m2, cfg):
                raise TokenLimitError("Limites de tokens excedidos durante exploração, possível não-limitabilidade.")

            mk2 = _marking_key(m2)
            outs.append((t, mk2))

            if mk2 not in pred:
                pred[mk2] = []
            pred[mk2].append(mk)

            if mk2 not in visited:
                visited.add(mk2)
                if len(visited) > cfg.max_states:
                    raise StateSpaceLimitError(f"Espaço de estados excedeu max_states={cfg.max_states}.")
                adj.setdefault(mk2, [])
                q.append((m2, mk2, depth + 1))

        adj[mk] = outs
        pred.setdefault(mk, [])

    if depth_cut:
        raise DepthLimitError(
            f"Exploração truncada por max_depth={cfg.max_depth}, resultados de OP/SL podem estar incompletos."
        )

    if fmk not in visited:
        raise ValueError("final_marking não é alcançável a partir do initial_marking.")

    return ReachabilityGraph(
        im_key=imk,
        fm_key=fmk,
        adj=adj,
        pred=pred,
        n_states=len(visited)
    )


def _reverse_reachable(pred: Dict[frozenset, List[frozenset]], target: frozenset) -> Set[frozenset]:
    can: Set[frozenset] = {target}
    q = deque([target])
    while q:
        x = q.popleft()
        for p in pred.get(x, []):
            if p not in can:
                can.add(p)
                q.append(p)
    return can


# ==============================
# OP and SL on reachability graph (per transition, no semantics calls)
# ==============================

def _reachable_without_transition_graph(
    imk: frozenset,
    fmk: frozenset,
    adj: Dict[frozenset, List[Tuple[Any, frozenset]]],
    forbidden: Any,
    restrict_to: Optional[Set[frozenset]] = None
) -> bool:
    """
    OP(t)=1 se existe execução completa im -> fm sem usar t.
    """
    if restrict_to is not None and imk not in restrict_to:
        return False

    seen: Set[frozenset] = {imk}
    q = deque([imk])

    while q:
        mk = q.popleft()
        if mk == fmk:
            return True
        for t, mk2 in adj.get(mk, []):
            if t == forbidden:
                continue
            if restrict_to is not None and mk2 not in restrict_to:
                continue
            if mk2 not in seen:
                seen.add(mk2)
                q.append(mk2)

    return False


def _reachable_with_consecutive_visible_pair_graph(
    imk: frozenset,
    fmk: frozenset,
    adj: Dict[frozenset, List[Tuple[Any, frozenset]]],
    target: Any,
    restrict_to: Optional[Set[frozenset]] = None
) -> bool:
    """
    SL(t)=1 se existe execução completa im -> fm tal que t ocorre duas vezes,
    sem nenhuma outra transição visível entre as duas ocorrências.
    Invisíveis entre as duas ocorrências são permitidas.
    """
    if restrict_to is not None and imk not in restrict_to:
        return False

    # phase:
    # 0: antes da 1a ocorrência de target
    # 1: após 1a ocorrência, só invisíveis ou target
    # 2: após 2a ocorrência, qualquer coisa até fm
    q = deque([(imk, 0)])
    seen: Set[Tuple[frozenset, int]] = {(imk, 0)}

    while q:
        mk, phase = q.popleft()

        if phase == 2 and mk == fmk:
            return True

        for t, mk2 in adj.get(mk, []):
            if restrict_to is not None and mk2 not in restrict_to:
                continue

            is_target = (t == target)
            is_visible = _is_visible(t)
            is_invisible = (not is_visible)

            if phase == 1 and is_visible and (not is_target):
                continue

            phase2 = phase
            if phase == 0:
                if is_target:
                    phase2 = 1
            elif phase == 1:
                if is_target:
                    phase2 = 2
                else:
                    if not is_invisible:
                        continue
                    phase2 = 1
            else:
                phase2 = 2

            key = (mk2, phase2)
            if key not in seen:
                seen.add(key)
                q.append((mk2, phase2))

    return False


# ==============================
# Structural proxies (cheap mode)
# ==============================

def _count_parallel_splits_structural(net: PetriNet, only_visible: bool = True) -> int:
    """
    Proxy de paralelismo: transição que produz tokens em múltiplos lugares.
    """
    c = 0
    for t in net.transitions:
        if only_visible and (not _is_visible(t)):
            continue
        postset = {a.target for a in getattr(t, "out_arcs", [])}
        if len(postset) > 1:
            c += 1
    return c


def _build_bipartite_digraph(net: PetriNet) -> nx.DiGraph:
    G = nx.DiGraph()
    for p in net.places:
        G.add_node(p)
    for t in net.transitions:
        G.add_node(t)

    arcs = getattr(net, "arcs", None)
    if arcs is not None:
        for a in arcs:
            G.add_edge(a.source, a.target)
        return G

    for t in net.transitions:
        for a in getattr(t, "in_arcs", []):
            G.add_edge(a.source, a.target)
        for a in getattr(t, "out_arcs", []):
            G.add_edge(a.source, a.target)
    return G


def _marking_sources(im: Marking, net: PetriNet) -> Set[Any]:
    src = {p for p, v in im.items() if v and int(v) > 0}
    if src:
        return src
    return {p for p in net.places if len(getattr(p, "in_arcs", [])) == 0}


def _marking_targets(fm: Optional[Marking], net: PetriNet) -> Set[Any]:
    if fm is not None:
        tgt = {p for p, v in fm.items() if v and int(v) > 0}
        if tgt:
            return tgt
    return {p for p in net.places if len(getattr(p, "out_arcs", [])) == 0}


def _reachable_nodes(G: nx.DiGraph, sources: Iterable[Any]) -> Set[Any]:
    seen: Set[Any] = set()
    q = deque()
    for s in sources:
        if s in G:
            seen.add(s)
            q.append(s)
    while q:
        u = q.popleft()
        for v in G.successors(u):
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen


def _structural_self_loop_like(
    G: nx.DiGraph,
    t_vis: Any,
    visible_set: Set[Any],
    allowed_nodes: Optional[Set[Any]] = None
) -> bool:
    """
    Proxy estrutural para SL(t): existe caminho dirigido t -> ... -> t que não passa
    por nenhuma outra transição visível (apenas lugares e transições invisíveis).
    """
    def ok(n: Any) -> bool:
        if allowed_nodes is not None and n not in allowed_nodes and n != t_vis:
            return False
        if n in visible_set and n != t_vis:
            return False
        return True

    seen: Set[Any] = set()
    q = deque()

    for v in G.successors(t_vis):
        if v == t_vis:
            return True
        if not ok(v):
            continue
        seen.add(v)
        q.append(v)

    while q:
        u = q.popleft()
        for v in G.successors(u):
            if v == t_vis:
                return True
            if v in seen:
                continue
            if not ok(v):
                continue
            seen.add(v)
            q.append(v)

    return False


def _op_sl_structural_proxies(
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Optional[Marking]
) -> Tuple[Dict[Any, int], Dict[Any, int], str]:
    """
    Proxies estruturais:
      OP(t)=1 se t não domina o super-sink (logo, é evitável).
      SL(t)=1 se existe auto-retorno estrutural de t sem outra visível no meio.

    Retorna também o método usado para OP ("dominators" ou "fallback_remove_node").
    """
    vis = _visible_transitions(net)
    op: Dict[Any, int] = {t: 0 for t in vis}
    sl: Dict[Any, int] = {t: 0 for t in vis}
    if not vis:
        return op, sl, "dominators"

    visible_set = set(vis)

    G = _build_bipartite_digraph(net)
    sources = _marking_sources(initial_marking, net)
    targets = _marking_targets(final_marking, net)

    reach_from_src = _reachable_nodes(G, sources)
    if not any(p in reach_from_src for p in targets):
        raise ValueError("Não há caminho estrutural das fontes aos alvos, rede não parece WF-net.")

    can_reach_tgt = _reachable_nodes(G.reverse(copy=False), targets)
    corridor = reach_from_src.intersection(can_reach_tgt)

    for t in vis:
        if t in corridor:
            sl[t] = 1 if _structural_self_loop_like(G, t, visible_set, allowed_nodes=corridor) else 0

    SRC = object()
    SNK = object()
    H = G.copy()
    H.add_node(SRC)
    H.add_node(SNK)

    for p in sources:
        if p in H:
            H.add_edge(SRC, p)
    for p in targets:
        if p in H:
            H.add_edge(p, SNK)

    op_method = "dominators"
    try:
        idom = nx.immediate_dominators(H, SRC)
        dom_chain: Set[Any] = set()
        cur: Any = SNK
        while True:
            dom_chain.add(cur)
            if cur == SRC:
                break
            cur = idom[cur]
        for t in vis:
            op[t] = 0 if (t in dom_chain) else 1
    except Exception:
        op_method = "fallback_remove_node"
        base_sources = list(sources)
        base_targets = list(targets)
        for t in vis:
            if t not in G:
                op[t] = 1
                continue
            H2 = G.copy()
            H2.remove_node(t)
            reach2 = _reachable_nodes(H2, base_sources)
            op[t] = 1 if any(p in reach2 for p in base_targets) else 0

    return op, sl, op_method


# ==============================
# Core (unchanged behavior: still computes Q0, Q1, Q2 together)
# ==============================

def _q_metrics_core(
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Optional[Marking] = None,
    *,
    mode: Literal["structural", "reachability"] = "structural",
    weights: QWeights = QWeights(),
    reach_cfg: ReachabilityConfig = ReachabilityConfig(),
    parallel_only_visible: bool = True,
    n0_policy: Literal["raise", "zeros", "ones"] = "raise",
    return_details: bool = True
) -> Union[Tuple[float, float, float], Tuple[float, float, float, Dict[str, Any]]]:
    vis = _visible_transitions(net)
    N = len(vis)

    if N == 0:
        if n0_policy == "raise":
            raise ValueError("N=0: não há transições visíveis (tarefas).")
        q = 1.0 if n0_policy == "ones" else 0.0
        details = {
            "mode": mode,
            "N_visible": 0,
            "OP": 0,
            "SL": 0,
            "PAR": _count_parallel_splits_structural(net, only_visible=parallel_only_visible),
            "B": 0.0,
            "B_univ": 0.0,
            "n0_policy": n0_policy,
        }
        return (q, q, q, details) if return_details else (q, q, q)

    PAR = _count_parallel_splits_structural(net, only_visible=parallel_only_visible)

    op_map: Dict[Any, int]
    sl_map: Dict[Any, int]
    extra_details: Dict[str, Any] = {}

    m = mode.lower().strip()
    if m == "structural":
        op_map, sl_map, op_method = _op_sl_structural_proxies(net, initial_marking, final_marking)
        extra_details["op_method"] = op_method

    elif m == "reachability":
        if final_marking is None:
            raise ValueError("mode='reachability' exige final_marking.")

        rg = _build_reachability_graph(net, initial_marking, final_marking, reach_cfg)
        can_reach_final = _reverse_reachable(rg.pred, rg.fm_key)

        op_map = {}
        sl_map = {}

        for t in vis:
            op_map[t] = 1 if _reachable_without_transition_graph(
                rg.im_key, rg.fm_key, rg.adj, forbidden=t, restrict_to=can_reach_final
            ) else 0

        for t in vis:
            sl_map[t] = 1 if _reachable_with_consecutive_visible_pair_graph(
                rg.im_key, rg.fm_key, rg.adj, target=t, restrict_to=can_reach_final
            ) else 0

        extra_details["reach_states"] = rg.n_states

    else:
        raise ValueError("mode deve ser 'structural' ou 'reachability'.")

    OP = int(sum(op_map.values()))
    SL = int(sum(sl_map.values()))

    # Q0
    q0 = 1.0 - ((OP + SL) / (2.0 * N))
    q0 = _clamp01(q0)

    # Badness
    B = float(N) + (weights.w_op * OP) + (weights.w_sl * SL) + (weights.w_par * PAR)

    # Upper bound for Q2 (bound simples para PAR)
    par_max = max(0, N - 1)
    B_univ = float(N) + (weights.w_op * N) + (weights.w_sl * N) + (weights.w_par * par_max)

    # Q1
    if N == 1:
        q1 = 1.0 if abs(B - 1.0) < 1e-12 else 0.0
    else:
        if B <= 1.0:
            q1 = 0.0
        else:
            q1 = math.log(float(N)) / math.log(B)
            q1 = _clamp01(q1)

    # Q2
    num_term = B - float(N) + 1.0
    den_term = B_univ - float(N) + 1.0
    if num_term <= 1.0:
        q2 = 1.0
    else:
        if den_term <= 1.0:
            q2 = 1.0
        else:
            q2 = 1.0 - (math.log(num_term) / math.log(den_term))
            q2 = _clamp01(q2)

    if not return_details:
        return (q0, q1, q2)

    details = {
        "mode": m,
        "N_visible": N,
        "OP": OP,
        "SL": SL,
        "PAR": PAR,
        "parallel_only_visible": parallel_only_visible,
        "w_op": weights.w_op,
        "w_sl": weights.w_sl,
        "w_par": weights.w_par,
        "B": B,
        "B_univ": B_univ,
        "op_by_transition": {_t_key(t): op_map[t] for t in vis},
        "sl_by_transition": {_t_key(t): sl_map[t] for t in vis},
        "reach_cfg": reach_cfg if m == "reachability" else None,
        "n0_policy": n0_policy,
    }
    details.update(extra_details)
    return (q0, q1, q2, details)
