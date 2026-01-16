from __future__ import annotations

from collections import Counter, defaultdict, deque
from metrics.simplicity.utils import simplicity_utils
from metrics.utils.output_range import *
from metrics.utils.quality_criteria import *
from pandas import DataFrame

from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments_petri_net
from pm4py.algo.conformance.alignments.petri_net.variants.state_equation_a_star import Parameters
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluation
from pm4py.objects.bpmn.obj import BPMN
from pm4py.objects.dfg.utils.dfg_utils import get_activities_from_dfg
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet, Marking

from typing import Any, Dict, Iterable, List, Set, Tuple, Union, Optional, Sequence

import copy
import heapq
import itertools
import math
import networkx as nx
import os
import pandas as pd
import pm4py
import re
import tempfile

LogLike = Union[
    pd.DataFrame,                 # event log em DataFrame
    Any,                          # PM4Py EventLog-like
    Iterable[Iterable[str]],      # traces já extraídos: lista de listas de activities
]

# Constants
MAX_ALIGN_TIME = 240

@classify(QualityCriteria.SIMPLICITY)
@output_range(SymbolicRange.ZERO_TO_ONE_INCLUSIVE)
def advanced_structural_appropriateness(
    net: PetriNet,
    im: Marking,
    fm: Optional[Marking] = None,
) -> float:
    """
    Advanced Structural Appropriateness (ASA) according to Rozinat (2008).

        a_S' = (|T| - (|T_DA| + |T_IR|)) / |T|

    where:
        T      = set of all transitions of the Petri net
        T_DA   = set of alternative duplicate tasks
        T_IR   = set of redundant invisible tasks

    Assumptions:
    - net, im, fm describe a bounded WF-net as in Rozinat (2008),
      i.e., sound workflow nets without dead transitions.
    - For non-WF nets, the result is a best-effort extension of the metric.

    References:
        - Rozinat, A. (2008). Conformance checking of processes based on
          monitoring real behavior.
        - Naderifar, V. et al. (2019). A review on conformance checking
          technique for the evaluation of process mining algorithms.
    """
    transitions = list(net.transitions)
    t_total = len(transitions)

    if t_total == 0:
        # Empty net: nothing to penalize
        return 1.0

    tir = simplicity_utils._count_redundant_invisible_tasks_strict(net, im, fm)
    alternative_duplicates = simplicity_utils._find_alternative_duplicate_tasks_marking_flow(net, im)
    tda = len(alternative_duplicates)

    penalty = tda + tir

    # Safety clamp (requested)
    if penalty > t_total:
        penalty = t_total

    asa = (t_total - penalty) / t_total
    return max(0.0, min(1.0, asa))

@classify(QualityCriteria.SIMPLICITY)
@output_range(SymbolicRange.ZERO_TO_ONE_INCLUSIVE)
def etm_simplicity_petri_net(
    petrinet: PetriNet,
    log: Union[DataFrame, EventLog],
    classifier_keys: Sequence[str] = ("concept:name",),
    classifier_sep: str = "||",
    *,
    return_details: bool = False,
) -> Union[float, Tuple[float, Dict[str, int]]]:
    """
    ETM-style simplicity, adapted to Petri nets, targeting the article behavior:

        Q = 1 - (A + B) / (C + D)

    D: #event classes in log
    C: #transitions in the net (visible + invisible)  [transition ~= leaf]
    A: duplicates among log-mappable visible labels
       + visible transitions not mappable to log classes
       + invisible transitions (tau)
    B: log classes missing from the model
    """
    from .utils.simplicity_utils import (
        EventClassifier,
        _event_classes_from_log,
        _is_invisible_transition_label,
    )
    classifier = EventClassifier(keys=classifier_keys, sep=classifier_sep)

    # D
    log_classes = _event_classes_from_log(log, classifier)
    D = len(log_classes)

    # Transitions split
    invisible_count = 0
    visible_labels_all = []

    for t in petrinet.transitions:
        if _is_invisible_transition_label(t.label):
            invisible_count += 1
        else:
            visible_labels_all.append(str(t.label).strip())

    # Map visible labels to log classes
    mapped_visible = [lab for lab in visible_labels_all if lab in log_classes]
    unmapped_visible_count = sum(1 for lab in visible_labels_all if lab not in log_classes)

    counts_mapped = Counter(mapped_visible)
    duplicates_mapped = sum(cnt - 1 for cnt in counts_mapped.values() if cnt > 1)

    # A and B
    A = duplicates_mapped + unmapped_visible_count + invisible_count
    B = len(log_classes - set(mapped_visible))

    # C: transitions count (closest analogue to tree leaves for the paper's example behavior)
    C = len(petrinet.transitions)

    denom = C + D
    if denom <= 0:
        raise ValueError("Invalid denominator (C + D). Check net and log inputs.")

    Q = 1.0 - (A + B) / float(denom)

    # By construction this should already be in [0,1], but keep a strict safety net.
    if Q < 0.0:
        Q = 0.0
    elif Q > 1.0:
        Q = 1.0

    if return_details:
        details = {
            "A_total": int(A),
            "A_duplicates_mapped": int(duplicates_mapped),
            "A_unmapped_visible": int(unmapped_visible_count),
            "A_invisible": int(invisible_count),
            "B_missing": int(B),
            "C_transitions": int(C),
            "D_log_classes": int(D),
            "num_visible_transitions": int(len(visible_labels_all)),
            "num_distinct_mapped_labels": int(len(set(mapped_visible))),
        }
        return Q, details

    return Q

@classify(QualityCriteria.SIMPLICITY)
@output_range(SymbolicRange.ZERO_TO_ONE_INCLUSIVE)
def df_structural_coverage(bpmn: BPMN, log_or_traces: LogLike) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """
    Simplicity adaptada do artigo para BPMN (PM4Py), no recorte do usuário.

    Fórmula:
      simplicity = 1 - (#da + #mo) / #no

    Definições (compromissos):
      - #no = |V| + |E| no grafo de controle expandido.
      - #da = redundância por rótulo de tarefa, soma(k-1) para cada label com multiplicidade k.
      - #mo = (nós não refletidos) + (arestas não refletidas), onde "refletido" é marcado por
        pares directly-follows do log projetados no grafo expandido, sem atravessar nós visíveis
        intermediários (tarefas, start, end).
      - Para cada par (a,b), marca a união de todos os caminhos mínimos entre qualquer nó com
        rótulo a e qualquer nó com rótulo b, não um caminho arbitrário.

    Expansão BPMN:
      - Nós reais considerados: StartEvent, EndEvent, Activity/Task, Gateway.
      - Arestas consideradas: SequenceFlow.
      - Gateways implícitos (convenção do usuário):
          * fan-in  >= 2 em nó NÃO-gateway => XOR-join implícito: VJ@id -> nó
          * fan-out >= 2 em nó NÃO-gateway => AND-split implícito: nó -> VS@id
          * se ambos, ambos são materializados.
      - Gateway explícito misto (fan-in >= 2 e fan-out >= 2):
          substitui o gateway por dois operadores: GJ@id -> GS@id,
          entrada do gateway aponta para GJ@id, saída do gateway sai de GS@id.

    Patch estrutural incorporado:
      - Trata corretamente pares directly-follows do tipo (A, A) ao calcular d_star via arestas
        (dist_from[u] + 1 + dist_to[v]), em vez de min(dist_from[target]).

    References:
        - Sungkono, H. A. et al. (2022). Graph-based Process Mining for
          Measuring Quality of Business Process Model.
    """

    activity_key = "concept:name"          	# para EventLog (dict-like events)
    case_id_col = None           		# para DataFrame, se quiser travar
    activity_col = None          		# para DataFrame, se quiser travar
    timestamp_col = None         		# para DataFrame, opcional
    lowercase_labels = False              	# normalização adicional
    include_start_end_pairs = True        	# inclui START->first e last->END nos DF pairs
    strict_duplicate_task_labels = False	# modo estrito: recusa labels duplicados em tasks
    clamp_0_1 = True                      	# clamp em [0, 1]
    return_details = False                	# retorna (simp, details)
    return_operator_sets = False	        # inclui conjuntos de nós/arestas refletidos e não refletidos
    details_max_pairs = 50                 	# limite de amostras em diagnostics

    START_LABEL = "START"
    END_LABEL = "END"

    # -------------------------
    # Normalização mínima
    # -------------------------
    def norm_label(s: Any) -> str:
        if s is None:
            return ""
        x = str(s)
        x = re.sub(r"\s+", " ", x.strip())
        if lowercase_labels:
            x = x.lower()
        return x

    # -------------------------
    # Coerção do log para traces
    # -------------------------
    def traces_from_dataframe(df: pd.DataFrame) -> List[List[str]]:
        ccands = [case_id_col] if case_id_col else [
            "case:concept:name", "case_id", "case", "CaseID", "Case Id"
        ]
        acands = [activity_col] if activity_col else [
            "concept:name", "activity", "Activity", "task", "Task"
        ]
        tcands = [timestamp_col] if timestamp_col else [
            "time:timestamp", "timestamp", "Timestamp", "time"
        ]

        ccol = next((c for c in ccands if c and c in df.columns), None)
        acol = next((c for c in acands if c and c in df.columns), None)
        tcol = next((c for c in tcands if c and c in df.columns), None)

        if ccol is None or acol is None:
            raise ValueError(
                "DataFrame precisa ter colunas de caso e de atividade, "
                "por exemplo 'case:concept:name' e 'concept:name'."
            )

        dfx = df.sort_values([ccol, tcol], kind="mergesort") if tcol is not None else df

        out: List[List[str]] = []
        for _, g in dfx.groupby(ccol, sort=False):
            seq: List[str] = []
            for x in g[acol].tolist():
                if pd.isna(x):
                    continue
                a = norm_label(x)
                if a:
                    seq.append(a)
            out.append(seq)
        return out

    def traces_from_eventlog(eventlog: Any) -> List[List[str]]:
        out: List[List[str]] = []
        for trace in eventlog:
            seq: List[str] = []
            for ev in trace:
                if isinstance(ev, dict) and activity_key in ev:
                    a = norm_label(ev.get(activity_key))
                    if a:
                        seq.append(a)
            out.append(seq)
        return out

    def coerce_traces(x: LogLike) -> List[List[str]]:
        if isinstance(x, pd.DataFrame):
            return traces_from_dataframe(x)

        if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
            it = iter(x)
            first = next(it, None)
            if first is None:
                return []

            def chain_first() -> Iterable[Any]:
                yield first
                for y in it:
                    yield y

            # EventLog-like: iterable de traces, trace é iterable de dicts
            if hasattr(first, "__iter__") and not isinstance(first, (str, bytes, dict)):
                first_ev = next(iter(first), None)
                if isinstance(first_ev, dict):
                    return traces_from_eventlog(chain_first())

                # traces explícitos: iterable de strings
                if isinstance(first_ev, (str, bytes)) or first_ev is None:
                    out: List[List[str]] = []
                    for tr in chain_first():
                        seq2: List[str] = []
                        for a in tr:
                            lab = norm_label(a)
                            if lab:
                                seq2.append(lab)
                        out.append(seq2)
                    return out

            # fallback conservador
            out2: List[List[str]] = []
            for tr in chain_first():
                if hasattr(tr, "__iter__") and not isinstance(tr, (str, bytes, dict)):
                    seq3: List[str] = []
                    for a in tr:
                        lab = norm_label(a)
                        if lab:
                            seq3.append(lab)
                    out2.append(seq3)
            if out2:
                return out2

        raise ValueError("Formato de log_or_traces não reconhecido. Use DataFrame, EventLog, ou traces.")

    traces = coerce_traces(log_or_traces)
    num_events = sum(len(t) for t in traces)

    # -------------------------
    # Directly-follows (pares)
    # -------------------------
    def build_df_pairs(trs: List[List[str]]) -> Tuple[Set[Tuple[str, str]], Set[str]]:
        pairs: Set[Tuple[str, str]] = set()
        labs: Set[str] = set()

        if include_start_end_pairs:
            labs.add(START_LABEL)
            labs.add(END_LABEL)

        for t in trs:
            if not t:
                continue
            labs.update(t)

            if include_start_end_pairs:
                pairs.add((START_LABEL, t[0]))

            for i in range(len(t) - 1):
                pairs.add((t[i], t[i + 1]))

            if include_start_end_pairs:
                pairs.add((t[-1], END_LABEL))

        return pairs, labs

    df_pairs, log_labels = build_df_pairs(traces)

    # -------------------------
    # BPMN: tipos e utilitários
    # -------------------------
    def is_seq_flow(f: Any) -> bool:
        return isinstance(f, BPMN.SequenceFlow)

    def is_gateway(n: Any) -> bool:
        return isinstance(n, BPMN.Gateway)

    def is_start(n: Any) -> bool:
        return isinstance(n, BPMN.StartEvent)

    def is_end(n: Any) -> bool:
        return isinstance(n, BPMN.EndEvent)

    def is_activity(n: Any) -> bool:
        return isinstance(n, BPMN.Activity) and not isinstance(n, BPMN.SubProcess)

    def node_id(n: Any) -> str:
        return str(n.get_id()) if hasattr(n, "get_id") else repr(n)

    def activity_label(n: Any) -> str:
        if not hasattr(n, "get_name"):
            return ""
        return norm_label(n.get_name())

    def visible_label(n: Any) -> str:
        if is_start(n):
            return START_LABEL
        if is_end(n):
            return END_LABEL
        if is_activity(n):
            return activity_label(n)
        return ""

    # -------------------------
    # Seleção de nós e flows
    # -------------------------
    nodes_all = list(bpmn.get_nodes()) if hasattr(bpmn, "get_nodes") else []
    flows_all = list(bpmn.get_flows()) if hasattr(bpmn, "get_flows") else []

    nodes = [n for n in nodes_all if is_start(n) or is_end(n) or is_activity(n) or is_gateway(n)]
    flows = [f for f in flows_all if is_seq_flow(f)]

    # Filtra flows para o universo de nós selecionados
    node_set = set(nodes)
    flows = [f for f in flows if (f.get_source() in node_set and f.get_target() in node_set)]

    # -------------------------
    # Graus considerando só SequenceFlow
    # -------------------------
    def in_flows(n: Any) -> List[Any]:
        if not hasattr(n, "get_in_arcs"):
            return []
        return [f for f in n.get_in_arcs() if is_seq_flow(f)]

    def out_flows(n: Any) -> List[Any]:
        if not hasattr(n, "get_out_arcs"):
            return []
        return [f for f in n.get_out_arcs() if is_seq_flow(f)]

    indeg: Dict[Any, int] = {n: len(in_flows(n)) for n in nodes}
    outdeg: Dict[Any, int] = {n: len(out_flows(n)) for n in nodes}

    # -------------------------
    # Implícitos e gateways mistos
    # -------------------------
    has_impl_join: Dict[Any, bool] = {}
    has_impl_split: Dict[Any, bool] = {}
    is_mixed_gateway: Dict[Any, bool] = {}

    for n in nodes:
        if is_gateway(n):
            is_mixed_gateway[n] = (indeg[n] >= 2 and outdeg[n] >= 2)
            has_impl_join[n] = False
            has_impl_split[n] = False
        else:
            is_mixed_gateway[n] = False
            has_impl_join[n] = (indeg[n] >= 2)
            has_impl_split[n] = (outdeg[n] >= 2)

    def vj_key(n: Any) -> str:
        return f"VJ@{node_id(n)}"

    def vs_key(n: Any) -> str:
        return f"VS@{node_id(n)}"

    def gj_key(n: Any) -> str:
        return f"GJ@{node_id(n)}"

    def gs_key(n: Any) -> str:
        return f"GS@{node_id(n)}"

    def entry(n: Any) -> Any:
        if is_gateway(n) and is_mixed_gateway.get(n, False):
            return gj_key(n)
        if has_impl_join.get(n, False):
            return vj_key(n)
        return n

    def exit(n: Any) -> Any:
        if is_gateway(n) and is_mixed_gateway.get(n, False):
            return gs_key(n)
        if has_impl_split.get(n, False):
            return vs_key(n)
        return n

    # -------------------------
    # Grafo expandido: V, E, adj, radj
    # -------------------------
    V: Set[Any] = set()
    adj: Dict[Any, List[Tuple[Any, str]]] = defaultdict(list)
    radj: Dict[Any, List[Tuple[Any, str]]] = defaultdict(list)
    edges: Dict[str, Tuple[Any, Any]] = {}
    edge_type: Dict[str, str] = {}

    edge_counter = 0

    def add_node(x: Any) -> None:
        V.add(x)

    def add_edge(u: Any, v: Any, etype: str, hint: str) -> str:
        nonlocal edge_counter
        edge_counter += 1
        eid = f"{hint}:{edge_counter}"
        edges[eid] = (u, v)
        edge_type[eid] = etype
        adj[u].append((v, eid))
        radj[v].append((u, eid))
        return eid

    # Nós reais, com substituição de gateway misto
    for n in nodes:
        if is_gateway(n) and is_mixed_gateway.get(n, False):
            add_node(gj_key(n))
            add_node(gs_key(n))
        else:
            add_node(n)

        if has_impl_join.get(n, False):
            add_node(vj_key(n))
        if has_impl_split.get(n, False):
            add_node(vs_key(n))

    # Arestas internas
    for n in nodes:
        if has_impl_join.get(n, False):
            add_edge(vj_key(n), n, "internal_vj", f"E_VJ_{node_id(n)}")
        if has_impl_split.get(n, False):
            add_edge(n, vs_key(n), "internal_vs", f"E_VS_{node_id(n)}")
        if is_gateway(n) and is_mixed_gateway.get(n, False):
            add_edge(gj_key(n), gs_key(n), "internal_gm", f"E_GM_{node_id(n)}")

    # Arestas SequenceFlow reconectadas via entry/exit
    for f in flows:
        u = f.get_source()
        v = f.get_target()
        add_edge(exit(u), entry(v), "sequence_flow", f"E_FLOW_{f.get_id()}")

    # -------------------------
    # Visibilidade e mapeamento label -> nós visíveis
    # -------------------------
    is_visible_node: Dict[Any, bool] = {x: False for x in V}
    label_to_visible_nodes: Dict[str, Set[Any]] = defaultdict(set)

    for n in nodes:
        if is_activity(n) or is_start(n) or is_end(n):
            is_visible_node[n] = True
            lab = visible_label(n)
            if lab:
                label_to_visible_nodes[lab].add(n)

    # -------------------------
    # #da: duplicidade de rótulo de tarefa
    # -------------------------
    task_labels: List[str] = []
    for n in nodes:
        if is_activity(n):
            lab = activity_label(n)
            if lab:
                task_labels.append(lab)

    counts = Counter(task_labels)
    duplicates = {k: v for k, v in counts.items() if v > 1}

    if strict_duplicate_task_labels and duplicates:
        raise ValueError(
            "Modo estrito: há tarefas com rótulo duplicado no BPMN. "
            f"Duplicatas: {sorted(duplicates.items(), key=lambda x: (-x[1], x[0]))}"
        )

    da = sum(max(0, v - 1) for v in counts.values())

    # Diagnóstico: tarefas do modelo não vistas no log
    model_task_label_set = set(task_labels)
    unseen_task_labels = sorted(model_task_label_set - (log_labels - {START_LABEL, END_LABEL}))

    # -------------------------
    # União de todos os caminhos mínimos por par (a,b)
    #   PATCH: d_star por dist_from + 1 + dist_to em arestas (trata (A,A))
    # -------------------------
    def union_of_all_shortest_paths(
        sources: Set[Any],
        targets: Set[Any],
    ) -> Tuple[Set[str], Set[Any]]:
        if not sources or not targets:
            return set(), set()

        # BFS forward: dist_from
        dist_from: Dict[Any, int] = {}
        q = deque()
        for s in sources:
            dist_from[s] = 0
            q.append(s)

        while q:
            cur = q.popleft()
            dcur = dist_from[cur]
            for nxt, _eid in adj.get(cur, []):
                if nxt in dist_from:
                    continue
                # bloqueia nó visível intermediário
                if is_visible_node.get(nxt, False) and nxt not in targets:
                    continue
                dist_from[nxt] = dcur + 1
                q.append(nxt)

        # BFS reverse: dist_to
        dist_to: Dict[Any, int] = {}
        q2 = deque()
        for t in targets:
            dist_to[t] = 0
            q2.append(t)

        while q2:
            cur = q2.popleft()
            dcur = dist_to[cur]
            for prv, _eid in radj.get(cur, []):
                if prv in dist_to:
                    continue
                # bloqueia nó visível intermediário
                if is_visible_node.get(prv, False) and prv not in sources:
                    continue
                dist_to[prv] = dcur + 1
                q2.append(prv)

        # d_star: menor comprimento > 0 via alguma aresta u->v conectando as regiões
        d_star: Optional[int] = None
        for u, du in dist_from.items():
            for v, _eid in adj.get(u, []):
                if v not in dist_to:
                    continue
                cand = du + 1 + dist_to[v]
                if cand < 1:
                    continue
                if d_star is None or cand < d_star:
                    d_star = cand

        if d_star is None:
            return set(), set()

        ref_e: Set[str] = set()
        ref_n: Set[Any] = set()

        # Aresta u->v está em algum caminho mínimo se dist_from[u] + 1 + dist_to[v] == d_star
        for u, du in dist_from.items():
            if du >= d_star:
                continue
            for v, eid in adj.get(u, []):
                if v not in dist_to:
                    continue
                if du + 1 + dist_to[v] == d_star:
                    ref_e.add(eid)
                    ref_n.add(u)
                    ref_n.add(v)

        return ref_e, ref_n

    cache: Dict[Tuple[str, str], Tuple[Set[str], Set[Any]]] = {}

    reflected_edges: Set[str] = set()
    reflected_nodes: Set[Any] = set()

    # Estatísticas de mapeamento de pares
    pair_stats = {
        "total_pairs": len(df_pairs),
        "pairs_no_sources_or_targets": 0,
        "pairs_no_path": 0,
        "pairs_mapped": 0,
    }
    pairs_no_sources_or_targets: List[Tuple[str, str]] = []
    pairs_no_path: List[Tuple[str, str]] = []

    # Marca nós visíveis que aparecem no log como refletidos
    for lab in log_labels:
        reflected_nodes.update(label_to_visible_nodes.get(lab, set()))

    # Marca operadores usados para explicar pares directly-follows
    for a, b in df_pairs:
        srcs = label_to_visible_nodes.get(a, set())
        tgts = label_to_visible_nodes.get(b, set())

        if not srcs or not tgts:
            pair_stats["pairs_no_sources_or_targets"] += 1
            if len(pairs_no_sources_or_targets) < details_max_pairs:
                pairs_no_sources_or_targets.append((a, b))
            continue

        key = (a, b)
        if key not in cache:
            cache[key] = union_of_all_shortest_paths(srcs, tgts)

        e_set, n_set = cache[key]
        if not e_set and not n_set:
            pair_stats["pairs_no_path"] += 1
            if len(pairs_no_path) < details_max_pairs:
                pairs_no_path.append((a, b))
            continue

        pair_stats["pairs_mapped"] += 1
        reflected_edges.update(e_set)
        reflected_nodes.update(n_set)

    # -------------------------
    # #no, #mo, simplicidade
    # -------------------------
    no = len(V) + len(edges)
    unref_nodes = len(V) - len(reflected_nodes)
    unref_edges = len(edges) - len(reflected_edges)
    mo = unref_nodes + unref_edges

    if no <= 0:
        simp = 1.0
    else:
        simp = 1.0 - float(da + mo) / float(no)

    if clamp_0_1:
        simp = max(0.0, min(1.0, float(simp)))
    else:
        simp = float(simp)

    if not return_details:
        return simp

    # -------------------------
    # Auditoria: contagens por tipo
    # -------------------------
    def node_kind(x: Any) -> str:
        if isinstance(x, str):
            if x.startswith("VJ@"):
                return "implicit_join_node"
            if x.startswith("VS@"):
                return "implicit_split_node"
            if x.startswith("GJ@"):
                return "mixed_gateway_join_part"
            if x.startswith("GS@"):
                return "mixed_gateway_split_part"
            return "virtual_other"
        # nó real BPMN
        if is_start(x):
            return "start"
        if is_end(x):
            return "end"
        if is_activity(x):
            return "task"
        if is_gateway(x):
            return "gateway"
        return "other"

    def edge_kind(eid: str) -> str:
        return edge_type.get(eid, "unknown")

    node_total_by_kind: Dict[str, int] = defaultdict(int)
    node_ref_by_kind: Dict[str, int] = defaultdict(int)
    edge_total_by_kind: Dict[str, int] = defaultdict(int)
    edge_ref_by_kind: Dict[str, int] = defaultdict(int)

    for x in V:
        node_total_by_kind[node_kind(x)] += 1
    for x in reflected_nodes:
        node_ref_by_kind[node_kind(x)] += 1

    for eid in edges.keys():
        edge_total_by_kind[edge_kind(eid)] += 1
    for eid in reflected_edges:
        edge_ref_by_kind[edge_kind(eid)] += 1

    node_unref_by_kind = {k: node_total_by_kind[k] - node_ref_by_kind.get(k, 0) for k in node_total_by_kind}
    edge_unref_by_kind = {k: edge_total_by_kind[k] - edge_ref_by_kind.get(k, 0) for k in edge_total_by_kind}

    details: Dict[str, Any] = {
        "simplicity": simp,
        "da_redundant_activities": da,
        "mo_operators_not_reflected": mo,
        "no_number_of_operators": no,
        "graph_nodes": len(V),
        "graph_edges": len(edges),
        "reflected_nodes": len(reflected_nodes),
        "reflected_edges": len(reflected_edges),
        "unref_nodes": unref_nodes,
        "unref_edges": unref_edges,
        "df_pairs": len(df_pairs),
        "df_pair_stats": dict(pair_stats),
        "df_pairs_no_sources_or_targets_sample": pairs_no_sources_or_targets,
        "df_pairs_no_path_sample": pairs_no_path,
        "traces": len(traces),
        "events": num_events,
        "duplicate_task_labels": dict(sorted(duplicates.items(), key=lambda x: (-x[1], x[0]))),
        "unseen_task_labels_in_log": unseen_task_labels,
        "nodes_total_by_kind": dict(sorted(node_total_by_kind.items())),
        "nodes_reflected_by_kind": dict(sorted(node_ref_by_kind.items())),
        "nodes_unreflected_by_kind": dict(sorted(node_unref_by_kind.items())),
        "edges_total_by_kind": dict(sorted(edge_total_by_kind.items())),
        "edges_reflected_by_kind": dict(sorted(edge_ref_by_kind.items())),
        "edges_unreflected_by_kind": dict(sorted(edge_unref_by_kind.items())),
        "params": {
            "lowercase_labels": bool(lowercase_labels),
            "include_start_end_pairs": bool(include_start_end_pairs),
            "strict_duplicate_task_labels": bool(strict_duplicate_task_labels),
            "clamp_0_1": bool(clamp_0_1),
            "details_max_pairs": int(details_max_pairs),
        },
    }

    if return_operator_sets:
        details["operator_sets"] = {
            "reflected_nodes": list(reflected_nodes),
            "reflected_edges": list(reflected_edges),
            "unreflected_nodes": [x for x in V if x not in reflected_nodes],
            "unreflected_edges": [eid for eid in edges.keys() if eid not in reflected_edges],
        }

    #return simp, details
    return simp


@classify(QualityCriteria.SIMPLICITY)
@output_range(SymbolicRange.ZERO_TO_ONE_INCLUSIVE)
def ssenc(net:PetriNet, log:Union[DataFrame, EventLog]) -> float:
    """Calculates the Structural Simplicity by Entity Count (SSENC) metric.

    Args:
        net (PetriNet): The Petri net model to analyze.
        log (Union[DataFrame, EventLog]): The event log containing observed
            behavior.

    Returns:
        float: The SSENC score, where a higher value typically indicates
        greater simplicity.

    References:
        - Adaptado de: Burke, D., & van der Aalst, W. M. P. (2024). A chance for models to
          show their quality: Stochastic process model-log dimensions.
        - Mendling, J., & van der Aalst, W. M. P. (2007). Understanding the
          occurrence of errors in process models based on metrics.
    """
    num_places = len(net.places)
    num_transitions = len(net.transitions)

    #num_traces = log['case:concept:name'].nunique()
    # ALTERAÇÃO AQUI: Obtém as variantes únicas do log
    variants = pm4py.get_variants(log)
    num_variants = len(variants)

    if num_variants == 0:
        return 0.0

    aux = (num_places + num_transitions) / num_variants
    return max(1 - aux, 0)
#enddef

@classify(QualityCriteria.SIMPLICITY)
@output_range(SymbolicRange.ZERO_TO_ONE_INCLUSIVE)
def ssedc(net:PetriNet, log:Union[DataFrame, EventLog]) -> float:
    """Calculates the Structural Simplicity by Edge Count (SSEDC) metric.

    Args:
        net (PetriNet): The Petri net model to analyze.
        log (Union[DataFrame, EventLog]): The event log containing observed
            behavior.

    Returns:
        float: The SSEDC score, where a higher value typically indicates
        greater simplicity.

    References:
        - Adaptado: Burke, D., & van der Aalst, W. M. P. (2024). A chance for models to
          show their quality: Stochastic process model-log dimensions.
        - Mendling, J., & van der Aalst, W. M. P. (2007). Understanding the
          occurrence of errors in process models based on metrics.
    """
    num_arcs = len(net.arcs) # Edges
    
    #num_traces = log['case:concept:name'].nunique()
    # ALTERAÇÃO AQUI: Obtém as variantes únicas do log
    variants = pm4py.get_variants(log)
    num_variants = len(variants)

    if num_variants == 0:
        return 0.0

    aux = num_arcs / num_variants
    return max(1 - aux, 0)
#enddef


@classify(QualityCriteria.SIMPLICITY)
@output_range(SymbolicRange.ZERO_TO_ONE_INCLUSIVE)
def structuredness(file_path: str) -> Dict[str, Any]:
    """Calculates the structuredness of a BPMN model from a file.

    This function serves as a wrapper to an external Java tool that computes
    the structuredness metric. Structuredness measures the degree to which a
    process model adheres to well-formed, single-entry, single-exit (SESE)
    block structures. A highly structured model is generally easier to
    understand and maintain.

    Args:
        file_path (str): The local file path to the BPMN model to be analyzed.

    Returns:
        Dict[str, Any]: A dictionary containing the structuredness metric results
        parsed from the external tool's JSON output.

    References:
        - Mendling, J., & Neumann, G. (2009). Metrics for Process Models.
    """
    from metrics.utils.java import helper as java_helper
    struc = java_helper.calculate_structuredness(file_path)
    return struc
#enddef

@classify(QualityCriteria.SIMPLICITY)
@output_range(SymbolicRange.ZERO_TO_ONE_INCLUSIVE)
def simple_structural_appropriateness(net:PetriNet) -> float:
    """Calculates the Simple Structural Appropriateness of a Petri net.

    Args:
        net (PetriNet): The Petri net model to analyze.

    Returns:
        float: The simple structural appropriateness score.

    References:
        - Rozinat, A., & van der Aalst, W. M. P. (2008). Conformance checking
          of processes based on monitoring real behavior.
        - Rozinat, A., & van der Aalst, W. M. P. (2005). Conformance testing:
          Measuring the fit and appropriateness of event logs and process models.
    """
    # Tasks in Petri Net
    task_labels = [t.label for t in net.transitions if t.label is not None]
    task_labels_set = set(task_labels) # unique elements
    t = len(task_labels_set)
    num_places = len(net.places)
    num_transitions = len(net.transitions)
    n = num_places + num_transitions
    if n == 0:
        return 0.0
    #endif
    raw_score = ( t + 2 ) / n
    return min(1.0, raw_score)
#enddef


@classify(QualityCriteria.SIMPLICITY)
@output_range(SymbolicRange.ZERO_TO_ONE_INCLUSIVE)
def sequentiality_ratio(bpmn: BPMN) -> float:
    """Calculates the Sequentiality Ratio of a BPMN model.

    This metric quantifies the degree to which a process model consists of
    sequential flows versus parallel or choice structures. It is typically
    calculated as the ratio of activities that are part of a sequence to the
    total number of activities in the model.

    A higher score indicates a more sequential, less complex process flow.

    Args:
        bpmn (BPMN): The BPMN model to analyze.

    Returns:
        float: The sequentiality ratio score.

    References:
        - Mendling, J., & Neumann, G. (2009). Metrics for Process Models.
    """
    seq_cls = getattr(BPMN, "SequenceFlow", None)
    gw_cls = getattr(BPMN, "Gateway", None)
    if seq_cls is None or gw_cls is None:
        raise RuntimeError("PM4Py BPMN classes not found (SequenceFlow/Gateway).")

    def is_gateway(node: object) -> bool:
        return isinstance(node, gw_cls)

    flows = [f for f in bpmn.get_flows() if isinstance(f, seq_cls)]
    if not flows:
        return 0.0

    valid_flows = []
    valid_flow_ids = set()
    for f in flows:
        u = f.get_source()
        v = f.get_target()
        if u is None or v is None:
            continue
        valid_flows.append(f)
        valid_flow_ids.add(f.get_id())

    if not valid_flows:
        return 0.0

    nodes = list(bpmn.get_nodes()) if hasattr(bpmn, "get_nodes") else []
    nodes = [n for n in nodes if n is not None]

    indeg = defaultdict(int)
    outdeg = defaultdict(int)

    if nodes:
        for n in nodes:
            in_sf = [
                a for a in n.get_in_arcs()
                if isinstance(a, seq_cls) and a.get_id() in valid_flow_ids
            ]
            out_sf = [
                a for a in n.get_out_arcs()
                if isinstance(a, seq_cls) and a.get_id() in valid_flow_ids
            ]
            indeg[n] = len(in_sf)
            outdeg[n] = len(out_sf)
    else:
        nodes_set = set()
        for f in valid_flows:
            u = f.get_source()
            v = f.get_target()
            nodes_set.add(u)
            nodes_set.add(v)
            outdeg[u] += 1
            indeg[v] += 1
        nodes = list(nodes_set)

    implicit_joins = {n for n in nodes if (not is_gateway(n) and indeg.get(n, 0) >= 2)}
    implicit_splits = {n for n in nodes if (not is_gateway(n) and outdeg.get(n, 0) >= 2)}

    mixed_gateways = {
        n for n in nodes
        if is_gateway(n) and indeg.get(n, 0) >= 2 and outdeg.get(n, 0) >= 2
    }

    total_arcs_expanded = (
        len(valid_flows)
        + len(implicit_joins)
        + len(implicit_splits)
        + len(mixed_gateways)
    )
    if total_arcs_expanded == 0:
        return 0.0

    sequential_arcs = 0
    for f in valid_flows:
        u = f.get_source()
        v = f.get_target()

        if is_gateway(u) or is_gateway(v):
            continue
        if u in implicit_splits:
            continue
        if v in implicit_joins:
            continue

        sequential_arcs += 1

    return sequential_arcs / total_arcs_expanded



# The following functions call an external Java program

@classify(QualityCriteria.SIMPLICITY)
@output_range(SymbolicRange.ZERO_TO_ONE_INCLUSIVE)
def separability(file_path: str) -> Dict[str, Any]:
    """Calculates the separability of a BPMN model from a file.

    This function acts as a wrapper for an external Java tool that computes
    the separability metric. Separability measures how well a process model
    can be decomposed into smaller, independent, and meaningful fragments.
    A model with high separability is often easier to understand and maintain.

    Args:
        file_path (str): The local file path to the BPMN model to be analyzed.

    Returns:
        Dict[str, Any]: A dictionary containing the separability metric results
        parsed from the external tool's JSON output.

    References:
        - Sanchez-Gonzalez, L., et al. (2010). Quality Assessment of Business
          Process Models based on Thresholds.
        - Mendling, J., & Neumann, G. (2009). Metrics for Process Models.
        - Mendling, J., et al. (2007). Error Metrics for Business Process Models.
    """
    from metrics.utils.java import helper as java_helper
    sep = java_helper.calculate_separability(file_path)
    return sep
#enddef

@classify(QualityCriteria.SIMPLICITY)
@output_range(SymbolicRange.ZERO_TO_ONE_INCLUSIVE)
def flow_hierarchy(bpmn: BPMN) -> float:
    """
    Flow hierarchy (h): fração de arestas que NÃO participam de ciclos em um grafo dirigido.
    Definição do artigo: "fraction of edges that do not participate in cycles in a directed graph". :contentReference[oaicite:6]{index=6}

    Adaptação BPMN (PM4Py):
      - Nós: StartEvent, EndEvent, Activity (exceto SubProcess), Gateway
      - Arestas: apenas SequenceFlow
      - Não materializa gateways implícitos nem separa gateways mistos, pois isso altera |E| sem respaldo na definição.
      - Se |E| == 0, retorna 1.0 (interpretação: ausência de arestas => nenhum ciclo).

    References:
        - Bantay, G. A. (2023). Frequent pattern mining-based log file partition
          for process mining.
    """
    allowed_types = (BPMN.StartEvent, BPMN.EndEvent, BPMN.Activity, BPMN.Gateway)
    root_pid = bpmn.get_process_id()

    # Candidatos (sem SubProcess)
    all_nodes = {
        n for n in bpmn.get_nodes()
        if isinstance(n, allowed_types) and not isinstance(n, BPMN.SubProcess)
    }

    # Tenta restringir ao processo raiz; se isso eliminar tudo, faz fallback (evita erro por DEFAULT_PROCESS)
    root_nodes = {n for n in all_nodes if n.get_process() == root_pid}
    nodes = root_nodes if root_nodes else all_nodes

    G = nx.MultiDiGraph()
    G.add_nodes_from(nodes)

    for f in bpmn.get_flows():
        if not isinstance(f, BPMN.SequenceFlow):
            continue
        u = f.get_source()
        v = f.get_target()
        if u in nodes and v in nodes:
            G.add_edge(u, v, key=f.get_id())

    m = G.number_of_edges()
    if m == 0:
        return 1.0

    cycle_edges = 0

    for comp in nx.strongly_connected_components(G):
        if len(comp) > 1:
            cycle_edges += G.subgraph(comp).number_of_edges()
        else:
            n = next(iter(comp))
            cycle_edges += G.number_of_edges(n, n)

    return (m - cycle_edges) / m

@classify(QualityCriteria.SIMPLICITY)
@output_range(SymbolicRange.ZERO_TO_ONE_INCLUSIVE)
def cross_connectivity(bpmn: BPMN) -> float:
    """
    Cross-Connectivity (CC) conforme Vanderfeesten et al., adaptada para BPMN (PM4Py).

    Base teorica:
      - N = T ∪ C (tarefas e conectores), e CC soma V(n1,n2) sobre n1,n2 ∈ N,
        normalizando por |N|(|N|-1).
      - d (grau) e o total de arcos incidentes (entradas + saidas).
      - W(a) = w(src(a)) * w(dest(a)), e V(n1,n2) e o maximo valor de caminho.

    Adaptacao BPMN (suas convencoes):
      - Considera apenas BPMN.SequenceFlow.
      - include_events=True inclui StartEvent e EndEvent como TASK (peso 1).
      - Gateways implicitos em nos NAO-gateway:
          fan-in  >= 2 => XOR-join implicito (VJ_<nid> -> nid)
          fan-out >= 2 => AND-split implicito (nid -> VS_<nid>)
          ambos => VJ_<nid> -> nid -> VS_<nid>
      - count_virtual_nodes:
          True  => pares (n1,n2) entre nos reais + virtuais (leitura direta de N = T ∪ C).
          False => pares (n1,n2) apenas entre nos reais, virtuais entram so nos caminhos
                   (boa opcao se voce quer comparabilidade so entre elementos BPMN "reais").
      - count_parallel_flows_in_degree:
          True  => graus contam multiplicidade de SequenceFlows (mais fiel a "arcos" como entidades).
          False => graus contam apenas pares (u,v) unicos.

    Retorna CC em [0, 1].

    References:
        - Vanderfeesten, I., et al. (2008). On a Quest for Good Process Models:
          The Cross-Connectivity Metric.
    """
    include_events = True
    count_virtual_nodes = True
    count_parallel_flows_in_degree = True
    nodes, edges_mult, kind, _ = simplicity_utils._build_augmented_graph(bpmn)
    if len(nodes) < 2:
        return 0.0

    if count_virtual_nodes:
        eval_nodes = set(nodes)
    else:
        eval_nodes = {nid for nid in nodes if not simplicity_utils._is_virtual(nid)}

    if len(eval_nodes) < 2:
        return 0.0

    indeg, outdeg = simplicity_utils._compute_degrees(nodes, edges_mult, deduplicate=not count_parallel_flows_in_degree)
    w = simplicity_utils._compute_node_weights(nodes, indeg, outdeg, kind)

    adjacency, arc_cost = simplicity_utils._build_adjacency_and_costs(nodes, edges_mult, w)

    node_list = sorted(eval_nodes)
    targets = set(node_list)

    total_V = 0.0
    for src in node_list:
        dist = simplicity_utils._dijkstra_shortest_paths(src, adjacency, arc_cost, targets=targets)
        for tgt in node_list:
            if tgt == src:
                continue
            d_ = dist.get(tgt)
            total_V += 0.0 if d_ is None else math.exp(-d_)

    denom = len(node_list) * (len(node_list) - 1)
    cc = total_V / float(denom)

    if cc < 0.0:
        return 0.0
    if cc > 1.0:
        return 1.0
    return float(cc)

@classify(QualityCriteria.SIMPLICITY)
@output_range(SymbolicRange.ZERO_TO_ONE_INCLUSIVE)
def global_efficiency(bpmn: BPMN) -> float:
    """
    Global efficiency E(G) (Latora and Marchiori), adaptada para BPMN (PM4Py).

    Definicao:
      E(G) = (1 / (N * (N - 1))) * sum_{i!=j} 1/d_ij
      com 1/d_ij = 0 quando nao existe caminho entre i e j.

    Decisoes fixas desta adaptacao:
      1) Distancias d_ij sao medidas entre os IDs reais dos nos pareados,
         nunca entre entry_map/exit_map (para nao pular o custo dos virtuais).
      2) Distancias sao calculadas na projecao nao dirigida do grafo.
      3) Pares incluem todos os nos BPMN reais no escopo: StartEvent, EndEvent,
         Activity, e Gateway. Nos virtuais nunca entram como pares.

    Convencoes BPMN do usuario:
      - Considera apenas BPMN.SequenceFlow.
      - Gateway implicito (apenas em nao gateway):
          fan-in  >= 2 => XOR-join implicito: VJOIN -> no
          fan-out >= 2 => AND-split implicito: no -> VSPLIT
          se ambos, ambos sao adicionados.
      - Gateway explicito duplo (gateway com fan-in >= 2 e fan-out >= 2):
          adiciona split virtual apos o gateway: gateway -> VSPLIT

    Retorno:
      - float em [0.0, 1.0], com pares inalcançaveis contribuindo 0.

    References:
        - Latora, V., & Marchiori, M. (2001). Efficient behavior of small-world
          networks.
        - Bantay, G. A. (2023). Frequent pattern mining-based log file partition
          for process mining.
    """
    include_implicit_gateways = True
    include_double_gateways = True
    restrict_to_main_process = True
    treat_as_undirected = True

    def _is_seq_flow(arc) -> bool:
        return isinstance(arc, BPMN.SequenceFlow)

    def _is_gateway(node) -> bool:
        return isinstance(node, BPMN.Gateway)

    def _is_pair_node(node) -> bool:
        return isinstance(node, (BPMN.StartEvent, BPMN.EndEvent, BPMN.Activity, BPMN.Gateway))

    def _id(node) -> str:
        return str(node.get_id())

    all_nodes = list(bpmn.get_nodes())

    if restrict_to_main_process:
        pid = bpmn.get_process_id()
        nodes = [n for n in all_nodes if n.get_process() == pid]
    else:
        nodes = all_nodes

    if len(nodes) < 2:
        return 0.0

    node_set = set(nodes)

    pair_nodes = [n for n in nodes if _is_pair_node(n)]
    if len(pair_nodes) < 2:
        return 0.0

    indeg = {}
    outdeg = {}
    for n in nodes:
        indeg[n] = sum(
            1
            for a in n.get_in_arcs()
            if _is_seq_flow(a) and a.get_source() in node_set
        )
        outdeg[n] = sum(
            1
            for a in n.get_out_arcs()
            if _is_seq_flow(a) and a.get_target() in node_set
        )

    G = nx.DiGraph()
    entry_map = {}
    exit_map = {}

    for n in nodes:
        nid = _id(n)
        G.add_node(nid, kind="element", type=type(n).__name__)

        is_gw = _is_gateway(n)
        din = indeg[n]
        dout = outdeg[n]

        if include_implicit_gateways and (not is_gw) and din >= 2:
            vj = f"__VJOIN__{nid}"
            G.add_node(vj, kind="virtual_join", owner=nid)
            G.add_edge(vj, nid)
            entry_map[n] = vj
        else:
            entry_map[n] = nid

        needs_split = False
        if include_implicit_gateways and (not is_gw) and dout >= 2:
            needs_split = True
        if include_double_gateways and is_gw and din >= 2 and dout >= 2:
            needs_split = True

        if needs_split:
            vs = f"__VSPLIT__{nid}"
            G.add_node(vs, kind="virtual_split", owner=nid)
            G.add_edge(nid, vs)
            exit_map[n] = vs
        else:
            exit_map[n] = nid

    for src in nodes:
        u = exit_map[src]
        for arc in src.get_out_arcs():
            if not _is_seq_flow(arc):
                continue
            tgt = arc.get_target()
            if tgt not in node_set:
                continue
            v = entry_map[tgt]
            if u != v:
                G.add_edge(u, v)

    H = G.to_undirected(as_view=False) if treat_as_undirected else G

    N = len(pair_nodes)
    denom = N * (N - 1)
    if denom == 0:
        return 0.0

    total = 0.0
    for src in pair_nodes:
        src_id = _id(src)
        dist = nx.single_source_shortest_path_length(H, src_id)
        for tgt in pair_nodes:
            if tgt is src:
                continue
            tgt_id = _id(tgt)
            d = dist.get(tgt_id)
            if d is None or d <= 0:
                continue
            total += 1.0 / d

    return total / denom

@classify(QualityCriteria.SIMPLICITY)
@output_range(SymbolicRange.ZERO_TO_ONE_INCLUSIVE)
def cyclicity(bpmn: BPMN) -> float:
    """Calculates the cyclicity of a BPMN.

    Args:
        

    Returns:
        float: The cyclicity score.

    References:
        - Mendling, J., & Neumann, G. (2009). Metrics for Process Models.
    """    
    pid = bpmn.get_process_id()
    all_nodes = list(bpmn.get_nodes())

    scoped_nodes = [n for n in all_nodes if hasattr(n, "get_process") and n.get_process() == pid]
    if not scoped_nodes:
        scoped_nodes = all_nodes
    if not scoped_nodes:
        return 0.0

    scoped_set = set(scoped_nodes)

    flows = []
    for f in bpmn.get_flows():
        if isinstance(f, BPMN.SequenceFlow):
            s = f.get_source()
            t = f.get_target()
            if s in scoped_set and t in scoped_set:
                flows.append((s, t))

    in_deg = {n: 0 for n in scoped_nodes}
    out_deg = {n: 0 for n in scoped_nodes}
    for s, t in flows:
        out_deg[s] += 1
        in_deg[t] += 1

    G = nx.DiGraph()

    def ij(n): return ("IJ", n.get_id())  # implicit XOR-join
    def is_(n): return ("IS", n.get_id())  # implicit AND-split
    def gj(n): return ("GJ", n.get_id())  # gateway join part
    def gs(n): return ("GS", n.get_id())  # gateway split part

    entry = {}
    exit_ = {}

    for n in scoped_nodes:
        indeg = in_deg[n]
        outdeg = out_deg[n]

        if isinstance(n, BPMN.Gateway):
            if indeg >= 2 and outdeg >= 2:
                j = gj(n)
                s = gs(n)
                G.add_node(j)
                G.add_node(s)
                G.add_edge(j, s)
                entry[n] = j
                exit_[n] = s
            else:
                G.add_node(n)
                entry[n] = n
                exit_[n] = n
        else:
            G.add_node(n)

            has_ij = indeg >= 2
            has_is = outdeg >= 2

            if has_ij:
                G.add_node(ij(n))
                G.add_edge(ij(n), n)

            if has_is:
                G.add_node(is_(n))
                G.add_edge(n, is_(n))

            entry[n] = ij(n) if has_ij else n
            exit_[n] = is_(n) if has_is else n

    for s, t in flows:
        G.add_edge(exit_[s], entry[t])

    total_nodes = G.number_of_nodes()
    if total_nodes == 0:
        return 0.0

    nodes_in_cycles = set()
    for comp in nx.strongly_connected_components(G):
        if len(comp) > 1:
            nodes_in_cycles.update(comp)
        else:
            (u,) = tuple(comp)
            if G.has_edge(u, u):
                nodes_in_cycles.add(u)

    return len(nodes_in_cycles) / total_nodes

@classify(QualityCriteria.SIMPLICITY)
@output_range(SymbolicRange.ZERO_TO_ONE_INCLUSIVE)
def density(bpmn: BPMN) -> float:
    """
    Density (Delta):
      Δ(G) = |A| / (|N| * (|N| - 1)), para grafo dirigido sem self-loop.

    Adaptação BPMN (PM4Py), conforme suas convenções:
      - Considera somente SequenceFlow.
      - Implícitos:
          * Event/Activity com fan-in  >= 2 => +1 nó e +1 arco (XOR-join implícito)
          * Event/Activity com fan-out >= 2 => +1 nó e +1 arco (AND-split implícito)
          * se ambos, soma ambos (duplo no mesmo elemento)
      - Duplo explícito:
          * Gateway com fan-in >= 2 e fan-out >= 2 => +1 nó e +1 arco

    Para coerência com o denominador, ignora self-loop (u->u) e colapsa arestas paralelas por (source, target).

    References:
        - Mendling, J., & Neumann, G. (2009). Metrics for Process Models.
    """

    pid = bpmn.get_process_id()

    allowed_nodes = (BPMN.Event, BPMN.Activity, BPMN.Gateway)
    node_set = {
        n for n in bpmn.get_nodes()
        if isinstance(n, allowed_nodes)
        and hasattr(n, "get_process")
        and n.get_process() == pid
    }

    edge_pairs = set()
    for f in bpmn.get_flows():
        if not isinstance(f, BPMN.SequenceFlow):
            continue
        s = f.get_source()
        t = f.get_target()
        if s in node_set and t in node_set and s != t:
            edge_pairs.add((s, t))

    n_eff = len(node_set)
    if n_eff <= 1:
        return 0.0

    a_eff = len(edge_pairs)

    in_deg = {n: 0 for n in node_set}
    out_deg = {n: 0 for n in node_set}
    for (s, t) in edge_pairs:
        out_deg[s] += 1
        in_deg[t] += 1

    for n in node_set:
        if isinstance(n, BPMN.Gateway):
            if in_deg[n] >= 2 and out_deg[n] >= 2:
                n_eff += 1
                a_eff += 1
            continue

        if in_deg[n] >= 2:
            n_eff += 1
            a_eff += 1
        if out_deg[n] >= 2:
            n_eff += 1
            a_eff += 1

    return a_eff / (n_eff * (n_eff - 1))

@classify(QualityCriteria.SIMPLICITY)
@output_range(SymbolicRange.ZERO_TO_ONE_INCLUSIVE)
def connector_heterogeneity(bpmn: BPMN) -> float:
    """
    Connector Heterogeneity (CH), base 3:
      CH(G) = - Σ_{l∈{AND,XOR,OR}} p(l) * log_3(p(l)),
      p(l) = |C_l| / |C|.

    BPMN adaptation, fixed conventions:
      - Explicit gateways:
        ParallelGateway -> AND, ExclusiveGateway -> XOR, InclusiveGateway -> OR
        Count by role:
          if fan-out >= 2: +1 (split occurrence)
          if fan-in  >= 2: +1 (join occurrence)
        If both, count 2 (join + split on the same element).
        Ignore degenerate gateways (fan-in < 2 and fan-out < 2).

      - Implicit gateways (Activity/Event, non-gateway nodes):
        fan-out >= 2: +1 implicit AND-split
        fan-in  >= 2: +1 implicit XOR-join
        If both, count 2.

    Notes:
      - Counts only SequenceFlow arcs (control-flow), ignoring MessageFlow/Association.
      - Assumes a single main process; filters nodes by bpmn.get_process_id() when possible.

    References:
        - Mendling, J., & Neumann, G. (2009). Metrics for Process Models.
    """
    if bpmn is None:
        raise ValueError("bpmn must not be None.")

    def _seq_in_arcs(node):
        arcs = node.get_in_arcs() if hasattr(node, "get_in_arcs") else getattr(node, "in_arcs", []) or []
        return [a for a in arcs if isinstance(a, BPMN.SequenceFlow)]

    def _seq_out_arcs(node):
        arcs = node.get_out_arcs() if hasattr(node, "get_out_arcs") else getattr(node, "out_arcs", []) or []
        return [a for a in arcs if isinstance(a, BPMN.SequenceFlow)]

    def fan_in(node) -> int:
        return len(_seq_in_arcs(node))

    def fan_out(node) -> int:
        return len(_seq_out_arcs(node))

    nodes = list(bpmn.get_nodes() or [])

    # If process information is consistent, keep only the main process nodes
    if hasattr(bpmn, "get_process_id"):
        pid = bpmn.get_process_id()
        filtered = []
        for n in nodes:
            if hasattr(n, "get_process"):
                if n.get_process() == pid:
                    filtered.append(n)
            else:
                filtered.append(n)
        nodes = filtered

    counts = {"AND": 0, "XOR": 0, "OR": 0}

    gateway_classes = (BPMN.ParallelGateway, BPMN.ExclusiveGateway, BPMN.InclusiveGateway)

    # 1) Explicit gateways, counted by role (join and split occurrences)
    for n in nodes:
        if not isinstance(n, gateway_classes):
            continue

        in_deg = fan_in(n)
        out_deg = fan_out(n)

        if in_deg < 2 and out_deg < 2:
            continue

        if isinstance(n, BPMN.ParallelGateway):
            typ = "AND"
        elif isinstance(n, BPMN.ExclusiveGateway):
            typ = "XOR"
        else:
            typ = "OR"

        if out_deg >= 2:
            counts[typ] += 1
        if in_deg >= 2:
            counts[typ] += 1

    # 2) Implicit gateways (Activity/Event only), excluding gateway nodes
    for n in nodes:
        if isinstance(n, BPMN.Gateway):
            continue
        if not isinstance(n, (BPMN.Activity, BPMN.Event)):
            continue

        in_deg = fan_in(n)
        out_deg = fan_out(n)

        if out_deg >= 2:
            counts["AND"] += 1
        if in_deg >= 2:
            counts["XOR"] += 1

    total = counts["AND"] + counts["XOR"] + counts["OR"]
    if total == 0:
        return 0.0

    ln3 = math.log(3.0)
    ch = 0.0
    for k in ("AND", "XOR", "OR"):
        c = counts[k]
        if c == 0:
            continue
        p = c / total
        ch -= p * (math.log(p) / ln3)

    # Numerical hygiene
    if abs(ch) < 1e-12:
        ch = 0.0
    if ch > 1.0 and (ch - 1.0) < 1e-12:
        ch = 1.0

    return ch

@classify(QualityCriteria.SIMPLICITY)
@output_range(SymbolicRange.ZERO_TO_ONE_INCLUSIVE)
def coupling(bpmn: BPMN) -> float:
    """
    Coupling (CP) do artigo, estendido para BPMN (PM4Py), com convenções fixas:

    - T: atividades do processo principal (BPMN.Activity), excluindo BPMN.SubProcess.
    - Conectores explícitos:
        AND (ParallelGateway): 1
        XOR (ExclusiveGateway): 1/(m*n)
        OR  (InclusiveGateway): 1/k + ((k-1)/k)*(1/(m*n)), k=(2^m-1)(2^n-1)
      (m = fan-in, n = fan-out).
    - Implícitos (convenção do usuário):
        split implícito em nó não-gateway com fan-out >= 2: AND, peso 1
        join  implícito em nó não-gateway com fan-in  >= 2: XOR-join com n=1, peso 1/m
    - Gateways duplos: emergem naturalmente via (m,n) no gateway, ou (join implícito * split implícito) em nó não-gateway.
    - Cadeias de gateways e eventos: permitidas (extensão para BPMN), usando o caminho de maior peso (produto de fatores).
    - Não atravessa nenhuma atividade intermediária (inclui SubProcess).
    - Se houver múltiplos caminhos sem atividades intermediárias entre t1 e t2, usa o maior peso.

    References:
        - Cardoso, J. (2015). Computing Coupling for Business Process Models.
    """

    pid = bpmn.get_process_id()
    nodes = [
        n for n in bpmn.get_nodes()
        if hasattr(n, "get_process") and n.get_process() == pid
    ]
    if not nodes:
        return 0.0
    node_set = set(nodes)

    tasks = [
        n for n in nodes
        if isinstance(n, BPMN.Activity) and not isinstance(n, BPMN.SubProcess)
    ]
    t_set = set(tasks)
    n_t = len(tasks)
    if n_t < 2:
        return 0.0

    # Bloqueia travessia por qualquer Activity (inclui SubProcess)
    activity_set = {n for n in nodes if isinstance(n, BPMN.Activity)}

    # Grafo simples: colapsa multiarestas para m e n representarem ramos, não arestas duplicadas
    out_adj = {n: set() for n in nodes}
    in_adj = {n: set() for n in nodes}

    for f in bpmn.get_flows():
        if not isinstance(f, BPMN.SequenceFlow):
            continue
        u = f.get_source()
        v = f.get_target()
        if u not in node_set or v not in node_set:
            continue
        if u == v:
            continue
        out_adj[u].add(v)
        in_adj[v].add(u)

    in_deg = {n: len(in_adj[n]) for n in nodes}
    out_deg = {n: len(out_adj[n]) for n in nodes}

    # Pesos do artigo
    def w_and(m: int, n: int) -> float:
        return 1.0 if m > 0 and n > 0 else 0.0

    def w_xor(m: int, n: int) -> float:
        return 1.0 / (m * n) if m > 0 and n > 0 else 0.0

    def w_or(m: int, n: int) -> float:
        if m <= 0 or n <= 0:
            return 0.0
        k = (2**m - 1) * (2**n - 1)
        return (1.0 / k) + ((k - 1.0) / k) * (1.0 / (m * n))

    def exit_factor(u) -> float:
        m = in_deg[u]
        n = out_deg[u]

        if isinstance(u, BPMN.ParallelGateway):
            return w_and(m, n)
        if isinstance(u, BPMN.ExclusiveGateway):
            return w_xor(m, n)
        if isinstance(u, BPMN.InclusiveGateway):
            return w_or(m, n)

        # split implícito (sua convenção): AND, peso 1
        if not isinstance(u, BPMN.Gateway) and n >= 2:
            return 1.0

        return 1.0

    def enter_factor(v) -> float:
        # Gateways explícitos já carregam (m,n) no exit_factor
        if isinstance(v, BPMN.Gateway):
            return 1.0
        m = in_deg[v]
        if m >= 2:
            return 1.0 / m  # XOR-join implícito com n=1
        return 1.0

    total = 0.0
    tie = itertools.count()

    for t1 in tasks:
        best_to_task = {}
        best_at = {t1: 1.0}

        # heap: (prioridade, desempate, nó)
        pq = [(-1.0, next(tie), t1)]

        while pq:
            neg_p_u, _, u = heapq.heappop(pq)
            p_u = -neg_p_u

            if p_u < best_at.get(u, 0.0):
                continue

            ef = exit_factor(u)
            if ef == 0.0:
                continue

            for v in out_adj[u]:
                p_v = p_u * ef * enter_factor(v)

                if v in t_set:
                    if v is not t1 and p_v > best_to_task.get(v, 0.0):
                        best_to_task[v] = p_v
                    continue

                if v in activity_set:
                    continue

                if p_v > best_at.get(v, 0.0):
                    best_at[v] = p_v
                    heapq.heappush(pq, (-p_v, next(tie), v))

        total += sum(best_to_task.values())

    return total / (n_t * (n_t - 1))

@classify(QualityCriteria.SIMPLICITY)
@output_range(SymbolicRange.ZERO_TO_ONE_INCLUSIVE)
def network_density(bpmn: BPMN) -> float:
    """
    Network density (di) em grafo dirigido simples:
      di = E / (N * (N - 1)), com di em [0, 1]

    Adaptação BPMN (PM4Py), com suas convenções fixas:
      - Nós explícitos: StartEvent, EndEvent, Task/Activity, Gateway (AND/XOR/OR).
      - Arestas explícitas: SequenceFlow, colapsando multiarestas (u, v) e removendo self-loop.
      - Gateways implícitos (apenas em nós NÃO-gateway):
          fan-out >= 2 => AND-split implícito: +1 nó, +1 aresta
          fan-in  >= 2 => XOR-join  implícito: +1 nó, +1 aresta
          se ambos: soma (+2 nós, +2 arestas)
      - Gateway explícito duplo:
          indeg >= 2 e outdeg >= 2 no mesmo gateway => join+split em série: +1 nó, +1 aresta
    References:
        - Bantay, G. A. (2023). Frequent pattern mining-based log file partition
          for process mining.
    """
    pid = bpmn.get_process_id()

    nodes = [
        n for n in bpmn.get_nodes()
        if hasattr(n, "get_process") and n.get_process() == pid
    ]
    if not nodes:
        return 0.0

    node_ids = {n.get_id() for n in nodes}

    edges = set()
    out_neighbors = {nid: set() for nid in node_ids}
    in_neighbors = {nid: set() for nid in node_ids}

    for f in bpmn.get_flows():
        if not isinstance(f, BPMN.SequenceFlow):
            continue

        u = f.get_source()
        v = f.get_target()

        if u.get_process() != pid or v.get_process() != pid:
            continue

        uid = u.get_id()
        vid = v.get_id()

        if uid not in node_ids or vid not in node_ids:
            continue
        if uid == vid:
            continue

        if (uid, vid) in edges:
            continue

        edges.add((uid, vid))
        out_neighbors[uid].add(vid)
        in_neighbors[vid].add(uid)

    n_exp = len(node_ids)
    e_exp = len(edges)

    implicit_splits = 0
    implicit_joins = 0
    explicit_doubles = 0

    for n in nodes:
        nid = n.get_id()
        indeg = len(in_neighbors.get(nid, set()))
        outdeg = len(out_neighbors.get(nid, set()))

        if isinstance(n, BPMN.Gateway):
            if indeg >= 2 and outdeg >= 2:
                explicit_doubles += 1
        else:
            if outdeg >= 2:
                implicit_splits += 1
            if indeg >= 2:
                implicit_joins += 1

    n_adj = n_exp + implicit_splits + implicit_joins + explicit_doubles
    if n_adj < 2:
        return 0.0

    e_adj = e_exp + implicit_splits + implicit_joins + explicit_doubles

    return e_adj / (n_adj * (n_adj - 1))

def structuredness_java(petrinet, initial_marking, final_marking):
    with tempfile.NamedTemporaryFile(suffix=".pnml", delete=False) as tmp:
        file_path = tmp.name
    try:
        pm4py.write_pnml(petrinet, initial_marking, final_marking, file_path)
        java_output = structuredness(file_path)
        value = float(java_output['structuredness'])
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
    return value

def separability_java(petrinet, initial_marking, final_marking):
    with tempfile.NamedTemporaryFile(suffix=".pnml", delete=False) as tmp:
        file_path = tmp.name
    try:
        pm4py.write_pnml(petrinet, initial_marking, final_marking, file_path)
        java_output = separability(file_path=file_path)
        value = float(java_output['separability'])
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
    return value

@classify(QualityCriteria.SIMPLICITY)
@output_range(SymbolicRange.ZERO_TO_ONE_INCLUSIVE)
def behavioral_specificity_q0(
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Optional[Marking] = None
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """Calculates the Q0 metric for a process tree.

    The Q0 metric is designed to evaluate the trade-off between the quality
    and accuracy of a discovered process model. It penalizes models that are
    overly complex or simplistic by analyzing the number of silent transitions
    and operators in the process tree.

    Args:
        tree (ProcessTree): The process tree model to analyze.

    Returns:
        float: The Q0 metric score.

    References:
        - Huang, Z., & Kumar, A. (2012). A study of quality and accuracy
          trade-offs in process mining.
    """
    out = simplicity_utils._q_metrics_core(
        net,
        initial_marking,
        final_marking,
        return_details=True,
    )
    q0_val, _, _, details = out  # type: ignore[misc]
    return q0_val

@classify(QualityCriteria.SIMPLICITY)
@output_range(SymbolicRange.ZERO_TO_ONE_INCLUSIVE)
def behavioral_specificity_q1(
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Optional[Marking] = None
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """Calculates the Q1 metric for a process tree.

    The Q1 metric evaluates the trade-off between the quality and accuracy
    of a discovered process model. It is designed to penalize models with
    a high "badness score", which reflects undesirable structural properties,
    relative to the size of the model.

    Args:
        tree (ProcessTree): The process tree model to analyze.

    Returns:
        float: The Q1 metric score.

    References:
        - Huang, Z., & Kumar, A. (2012). A study of quality and accuracy
          trade-offs in process mining.
    """
    out = simplicity_utils._q_metrics_core(
        net,
        initial_marking,
        final_marking,
        return_details=True,
    )
    _, q1_val, _, details = out  # type: ignore[misc]
    return q1_val

@classify(QualityCriteria.SIMPLICITY)
@output_range(SymbolicRange.ZERO_TO_ONE_INCLUSIVE)
def behavioral_specificity_q2(
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Optional[Marking] = None
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """Calculates the Q2 metric for a process tree.

    The Q2 metric evaluates the trade-off between the quality and accuracy
    of a discovered process model. It compares the "badness" of the actual
    discovered model against the "badness" of a universal model (one that
    allows all behavior), relative to the number of activities.

    Args:
        tree (ProcessTree): The process tree model to analyze.

    Returns:
        float: The Q2 metric score.

    References:
        - Huang, Z., & Kumar, A. (2012). A study of quality and accuracy
          trade-offs in process mining.
    """
    out = simplicity_utils._q_metrics_core(
        net,
        initial_marking,
        final_marking,
        return_details=True,
    )
    _, _, q2_val, details = out  # type: ignore[misc]
    return q2_val