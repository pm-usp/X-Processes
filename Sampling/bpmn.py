from __future__ import annotations
from collections import deque
from typing import Dict, List, Sequence, Set, Tuple, Optional
import copy
import numpy as np
import cloudpickle
from pm4py.objects.bpmn.obj import BPMN
import re
import unicodedata
import difflib


class ChromosomeConversionError(RuntimeError):
    pass


def bpmn_to_cromossome(
    bpmn_graph: BPMN,
    alphabet: Sequence[str],
    *,
    mutate_input: bool = False,
    verbose: bool = False,
    require_stable_ids: bool = True,
    enforce_start_to_end_path: bool = True,
    allow_self_loops: bool = True,
) -> np.ndarray:
    """
    Converte um BPMN (objeto BPMN da PM4Py) para o cromossomo do seu GA:

      - alphabet[0] == 'begin' e alphabet[-1] == 'end'
      - cromossomo shape = (n+1, n+1), n = len(alphabet)
      - A = crom[:n, :n] é adjacência imediata (0/1), incluindo begin/end
      - meta_split = crom[:n, -1] é tipo de split por nó (0/1/2/3, ou -1 para task ausente)
      - meta_join  = crom[-1, :n] é tipo de join por nó (0/1/2/3, ou -1 para task ausente)

    Políticas (estritas):
      - deduplicar flows paralelos por par (source, target), antes e durante o pré-processamento
      - remover gateways degenerados (unique in=1 e unique out=1) por bypass
          * exceção: se pred==succ e allow_self_loops=True, preservar (não remover) para não apagar semântica de loop
          * se pred==succ e allow_self_loops=False, falhar (loop não representável no seu encoding)
      - remover gateways órfãos (0 in ou 0 out) de forma destrutiva
      - gateway misto (unique in>1 e unique out>1) vira dois gateways em série: join -> split (mesmo tipo)
      - gateway implícito:
            * nó com 2+ saídas distintas (task ou start) -> inserir split AND (Parallel)
            * nó com 2+ entradas distintas (task ou end)  -> inserir join XOR (Exclusive)
      - intermediário que não seja gateway no caminho task->...->task:
            * falha
      - labels duplicados de task no BPMN:
            * falha
      - task no BPMN fora do alphabet:
            * falha
      - conectividade:
            * EndEvent alcançável do StartEvent
            * opcionalmente, toda task no BPMN precisa estar em algum caminho Start -> ... -> End
      - determinismo:
            * travessia ordenada por (tipo, name, id), pode exigir ids estáveis

    Self-loop (A[i,i]):
      - allow_self_loops=True mantém (apenas para tasks; begin/end continuam proibidos)
      - allow_self_loops=False falha se ocorrer
    """

    def _norm_label(v) -> str:
        s = unicodedata.normalize("NFKC", "" if v is None else str(v))
        s = s.replace("\u00a0", " ")  # NBSP
        s = re.sub(r"\s+", " ", s).strip()  # colapsa whitespace e strip
        return s

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    def _get_id_str(obj) -> str:
        if hasattr(obj, "get_id"):
            try:
                v = obj.get_id()
                return "" if v is None else str(v)
            except Exception:
                return ""
        return ""

    def _get_name_str(obj) -> str:
        if hasattr(obj, "get_name"):
            try:
                v = obj.get_name()
                return "" if v is None else str(v)
            except Exception:
                return ""
        return ""

    def fail(msg: str) -> None:
        log(f"[bpmn_to_cromossome_pm4py] ERRO: {msg}")
        raise ChromosomeConversionError(msg)

    def _require_id(kind: str, obj, label_hint: str = "") -> str:
        v = _get_id_str(obj)
        if v != "":
            return v
        if require_stable_ids:
            suffix = f" (label='{label_hint}')" if label_hint else ""
            fail(
                f"{kind} sem get_id() estável{suffix}. "
                f"Sem isso, a ordenação pode variar entre execuções. "
                f"Se você aceitar isso, chame com require_stable_ids=False."
            )
        return ""

    def node_key(node) -> Tuple[str, str, str]:
        nm_raw = _get_name_str(node)
        nm = _norm_label(nm_raw)
        nid = _require_id("Nó", node, label_hint=nm)
        return (type(node).__name__, nm, nid)

    def flow_key(flow) -> Tuple[Tuple[str, str, str], Tuple[str, str, str], str]:
        src = flow.get_source()
        tgt = flow.get_target()
        fid = _require_id("Flow", flow)
        return (node_key(src), node_key(tgt), fid)

    def out_arcs_sorted(node) -> List[object]:
        arcs = list(node.get_out_arcs())
        arcs.sort(key=flow_key)
        return arcs

    def in_arcs_sorted(node) -> List[object]:
        arcs = list(node.get_in_arcs())
        arcs.sort(key=flow_key)
        return arcs

    def out_targets_sorted(node) -> List[object]:
        return [f.get_target() for f in out_arcs_sorted(node)]

    def in_sources_sorted(node) -> List[object]:
        return [f.get_source() for f in in_arcs_sorted(node)]

    # =========
    # 0) Copiar ou mutar
    # =========
    if mutate_input:
        g = bpmn_graph
    else:
        try:
            g = copy.deepcopy(bpmn_graph)
        except Exception:
            g = cloudpickle.loads(cloudpickle.dumps(bpmn_graph))

    # =========
    # 1) Validar alphabet e preparar índices
    # =========
    alphabet_raw_list = ["" if x is None else str(x) for x in alphabet]
    alphabet_list = [_norm_label(x) for x in alphabet_raw_list]

    if len(alphabet_list) < 2:
        fail("alphabet inválido, precisa ter pelo menos ['begin', 'end'].")

    if alphabet_list[0] != "begin" or alphabet_list[-1] != "end":
        fail("alphabet precisa começar com 'begin' e terminar com 'end' (após normalização).")

    n = len(alphabet_list)

    idx_of: Dict[str, int] = {}
    raw_by_norm: Dict[str, str] = {}

    for i, (raw, norm) in enumerate(zip(alphabet_raw_list, alphabet_list)):
        if norm == "":
            fail(f"alphabet contém label vazio na posição {i} (raw={raw!r}).")
        if norm in idx_of:
            prev_raw = raw_by_norm.get(norm, "")
            fail(
                f"alphabet contém labels duplicados após normalização: {norm!r}. "
                f"Exemplos: {prev_raw!r} e {raw!r}."
            )
        idx_of[norm] = i
        raw_by_norm[norm] = raw

    # =========
    # 2) Helpers de tipos
    # =========
    def is_gateway(node) -> bool:
        return isinstance(node, (BPMN.ExclusiveGateway, BPMN.ParallelGateway, BPMN.InclusiveGateway))

    def is_task(node) -> bool:
        return isinstance(node, BPMN.Activity) and not isinstance(node, BPMN.SubProcess)

    def is_start(node) -> bool:
        return isinstance(node, BPMN.StartEvent)

    def is_end(node) -> bool:
        return isinstance(node, BPMN.EndEvent)

    def is_supported_main_node(node) -> bool:
        return is_start(node) or is_end(node) or is_task(node) or is_gateway(node)

    def gateway_type_value(node) -> int:
        if isinstance(node, BPMN.ExclusiveGateway):
            return 1
        if isinstance(node, BPMN.ParallelGateway):
            return 2
        if isinstance(node, BPMN.InclusiveGateway):
            return 3
        fail(f"Gateway não suportado: {type(node).__name__}")
        return 0  # unreachable

    def fmt_node(node) -> str:
        nm_raw = _get_name_str(node)
        nm = _norm_label(nm_raw)
        return f"type={type(node).__name__} id='{_get_id_str(node)}' name_raw={nm_raw!r} name_norm={nm!r}"

    def unique_out_targets(node) -> List[object]:
        seen: Set[object] = set()
        out: List[object] = []
        for f in out_arcs_sorted(node):
            tgt = f.get_target()
            if tgt in seen:
                continue
            seen.add(tgt)
            out.append(tgt)
        return out

    def unique_in_sources(node) -> List[object]:
        seen: Set[object] = set()
        out: List[object] = []
        for f in in_arcs_sorted(node):
            src = f.get_source()
            if src in seen:
                continue
            seen.add(src)
            out.append(src)
        return out

    def out_deg_unique(node) -> int:
        return len(unique_out_targets(node))

    def in_deg_unique(node) -> int:
        return len(unique_in_sources(node))

    def flow_exists(source, target) -> bool:
        for f in out_arcs_sorted(source):
            if f.get_target() == target:
                return True
        return False

    # =========
    # 3) Processo principal e sanidade de nós
    # =========
    process_id = g.get_process_id()
    main_nodes = [x for x in g.get_nodes() if x.get_process() == process_id]
    main_nodes_sorted = sorted(main_nodes, key=node_key)

    for node in main_nodes_sorted:
        if not is_supported_main_node(node):
            fail(f"Nó não suportado no processo principal: {fmt_node(node)}.")
        if isinstance(node, BPMN.SubProcess):
            fail(f"SubProcess não suportado no processo principal: {fmt_node(node)}.")

    starts = [x for x in main_nodes_sorted if is_start(x)]
    ends = [x for x in main_nodes_sorted if is_end(x)]

    if len(starts) != 1:
        fail(f"Esperado exatamente 1 StartEvent no processo principal, encontrado {len(starts)}.")
    if len(ends) != 1:
        fail(f"Esperado exatamente 1 EndEvent no processo principal, encontrado {len(ends)}.")

    start_event = sorted(starts, key=node_key)[0]
    end_event = sorted(ends, key=node_key)[0]

    # =========
    # 4) Mapear tasks por label e validar duplicatas e fora do alphabet
    # =========
    tasks = [node for node in main_nodes_sorted if is_task(node)]
    name_to_task: Dict[str, object] = {}

    for t in tasks:
        nm_raw = _get_name_str(t)
        name = _norm_label(nm_raw)
        if name == "":
            fail(f"Task sem label (name vazio) no processo principal: {fmt_node(t)}.")
        if name in ("begin", "end"):
            fail("BPMN contém task com label reservado 'begin' ou 'end' (após normalização).")
        if name in name_to_task:
            fail(f"Labels duplicados de task no BPMN após normalização: {name!r}.")
        name_to_task[name] = t

    for name in name_to_task.keys():
        if name not in idx_of:
            sugestoes = difflib.get_close_matches(name, list(idx_of.keys()), n=5, cutoff=0.82)
            extra = f" Sugestões: {sugestoes}" if sugestoes else ""
            fail(f"Task no BPMN não existe no alphabet: {name!r}.{extra}")

    # =========
    # 5) IDs únicos para nós e flows novos
    # =========
    used_ids: Set[str] = set()
    for node in g.get_nodes():
        nid = _get_id_str(node)
        if nid:
            used_ids.add(nid)
        for f in list(getattr(node, "get_out_arcs", lambda: [])()):
            fid = _get_id_str(f)
            if fid:
                used_ids.add(fid)

    counters = {"gw": 0, "flow": 0}

    def new_unique_id(prefix: str) -> str:
        while True:
            counters[prefix] += 1
            cand = f"{prefix}_{counters[prefix]}"
            if cand not in used_ids:
                used_ids.add(cand)
                return cand

    # =========
    # 6) Deduplicação de flows paralelos por (source, target)
    # =========
    def dedup_parallel_flows_once() -> bool:
        removed = 0
        sources = sorted([x for x in g.get_nodes() if x.get_process() == process_id], key=node_key)
        for src in sources:
            seen_targets: Set[object] = set()
            for f in out_arcs_sorted(src):
                tgt = f.get_target()
                if tgt in seen_targets:
                    g.remove_flow(f)
                    removed += 1
                else:
                    seen_targets.add(tgt)
        if removed > 0:
            log(f"[pre] dedup flows: removidos {removed} flows duplicados por (source,target).")
        return removed > 0

    # =========
    # 7) Pré-processamento estrutural (iterativo, determinístico, com unique-degree)
    # =========
    def remove_node_with_incident_flows(node) -> None:
        for f in list(in_arcs_sorted(node)):
            g.remove_flow(f)
        for f in list(out_arcs_sorted(node)):
            g.remove_flow(f)
        g.remove_node(node)

    def remove_orphan_gateways_once() -> bool:
        changed = False
        gateways = [node for node in g.get_nodes() if is_gateway(node) and node.get_process() == process_id]
        for gw in sorted(gateways, key=node_key):
            if len(in_arcs_sorted(gw)) == 0 or len(out_arcs_sorted(gw)) == 0:
                log(f"[pre] removendo gateway órfão: {fmt_node(gw)}")
                remove_node_with_incident_flows(gw)
                changed = True
        return changed

    def remove_degenerate_gateways_once() -> bool:
        changed = False
        gateways = [node for node in g.get_nodes() if is_gateway(node) and node.get_process() == process_id]
        for gw in sorted(gateways, key=node_key):
            in_srcs = unique_in_sources(gw)
            out_tgts = unique_out_targets(gw)

            if len(in_srcs) == 1 and len(out_tgts) == 1:
                pred = in_srcs[0]
                succ = out_tgts[0]

                if pred == succ:
                    if allow_self_loops:
                        continue
                    fail(
                        f"Estrutura induz self-loop via gateway degenerado (pred==succ) em {fmt_node(gw)}, "
                        f"mas allow_self_loops=False."
                    )

                if not flow_exists(pred, succ):
                    fid = new_unique_id("flow")
                    g.add_flow(BPMN.SequenceFlow(pred, succ, id=fid, process=process_id))

                log(f"[pre] bypass gateway degenerado: {fmt_node(gw)} pred=({fmt_node(pred)}) succ=({fmt_node(succ)})")
                remove_node_with_incident_flows(gw)
                changed = True
        return changed

    def make_gateway_of_same_type(original_gateway, new_id: str, *, direction) -> object:
        if isinstance(original_gateway, BPMN.ExclusiveGateway):
            return BPMN.ExclusiveGateway(id=new_id, process=process_id, gateway_direction=direction)
        if isinstance(original_gateway, BPMN.ParallelGateway):
            return BPMN.ParallelGateway(id=new_id, process=process_id, gateway_direction=direction)
        if isinstance(original_gateway, BPMN.InclusiveGateway):
            return BPMN.InclusiveGateway(id=new_id, process=process_id, gateway_direction=direction)
        fail(f"Gateway não suportado: {type(original_gateway).__name__}")

    def normalize_mixed_gateways_once() -> bool:
        changed = False
        gateways = [node for node in g.get_nodes() if is_gateway(node) and node.get_process() == process_id]
        for gw in sorted(gateways, key=node_key):
            indeg_u = in_deg_unique(gw)
            outdeg_u = out_deg_unique(gw)

            if indeg_u > 1 and outdeg_u > 1:
                log(f"[pre] normalizando gateway misto: {fmt_node(gw)} in_u={indeg_u} out_u={outdeg_u}")

                join_id = new_unique_id("gw")
                split_id = new_unique_id("gw")

                join_gw = make_gateway_of_same_type(gw, join_id, direction=BPMN.Gateway.Direction.CONVERGING)
                split_gw = make_gateway_of_same_type(gw, split_id, direction=BPMN.Gateway.Direction.DIVERGING)

                g.add_node(join_gw)
                g.add_node(split_gw)

                preds = unique_in_sources(gw)
                succs = unique_out_targets(gw)

                for f in list(in_arcs_sorted(gw)):
                    g.remove_flow(f)
                for f in list(out_arcs_sorted(gw)):
                    g.remove_flow(f)

                for pred in preds:
                    if not flow_exists(pred, join_gw):
                        fid = new_unique_id("flow")
                        g.add_flow(BPMN.SequenceFlow(pred, join_gw, id=fid, process=process_id))

                if not flow_exists(join_gw, split_gw):
                    fid = new_unique_id("flow")
                    g.add_flow(BPMN.SequenceFlow(join_gw, split_gw, id=fid, process=process_id))

                for succ in succs:
                    if not flow_exists(split_gw, succ):
                        fid = new_unique_id("flow")
                        g.add_flow(BPMN.SequenceFlow(split_gw, succ, id=fid, process=process_id))

                g.remove_node(gw)
                changed = True
        return changed

    def insert_implicit_splits_once() -> bool:
        changed = False
        candidates = [
            node
            for node in g.get_nodes()
            if node.get_process() == process_id and (is_task(node) or is_start(node))
        ]
        for node in sorted(candidates, key=node_key):
            targets_u = unique_out_targets(node)
            if len(targets_u) >= 2:
                gw_id = new_unique_id("gw")
                split_gw = BPMN.ParallelGateway(
                    id=gw_id,
                    process=process_id,
                    gateway_direction=BPMN.Gateway.Direction.DIVERGING,
                )
                g.add_node(split_gw)

                log(f"[pre] inserindo split AND implícito após {fmt_node(node)}")

                for f in list(out_arcs_sorted(node)):
                    g.remove_flow(f)

                fid = new_unique_id("flow")
                g.add_flow(BPMN.SequenceFlow(node, split_gw, id=fid, process=process_id))

                for tgt in targets_u:
                    fid = new_unique_id("flow")
                    g.add_flow(BPMN.SequenceFlow(split_gw, tgt, id=fid, process=process_id))

                changed = True
        return changed

    def insert_implicit_joins_once() -> bool:
        changed = False
        candidates = [
            node
            for node in g.get_nodes()
            if node.get_process() == process_id and (is_task(node) or is_end(node))
        ]
        for node in sorted(candidates, key=node_key):
            sources_u = unique_in_sources(node)
            if len(sources_u) >= 2:
                gw_id = new_unique_id("gw")
                join_gw = BPMN.ExclusiveGateway(
                    id=gw_id,
                    process=process_id,
                    gateway_direction=BPMN.Gateway.Direction.CONVERGING,
                )
                g.add_node(join_gw)

                log(f"[pre] inserindo join XOR implícito antes de {fmt_node(node)}")

                for f in list(in_arcs_sorted(node)):
                    g.remove_flow(f)

                for src in sources_u:
                    fid = new_unique_id("flow")
                    g.add_flow(BPMN.SequenceFlow(src, join_gw, id=fid, process=process_id))

                fid = new_unique_id("flow")
                g.add_flow(BPMN.SequenceFlow(join_gw, node, id=fid, process=process_id))

                changed = True
        return changed

    dedup_parallel_flows_once()

    guard = 0
    changed = True
    while changed:
        guard += 1
        if guard > 50:
            fail("Pré-processamento excedeu limite de iterações, possível ciclo de reescrita.")

        changed = False

        changed |= dedup_parallel_flows_once()

        changed |= remove_orphan_gateways_once()
        changed |= remove_degenerate_gateways_once()
        changed |= normalize_mixed_gateways_once()
        changed |= dedup_parallel_flows_once()

        changed |= insert_implicit_splits_once()
        changed |= insert_implicit_joins_once()
        changed |= dedup_parallel_flows_once()

        changed |= remove_orphan_gateways_once()
        changed |= remove_degenerate_gateways_once()
        changed |= normalize_mixed_gateways_once()
        changed |= dedup_parallel_flows_once()

    # =========
    # 8) Checagens estruturais no BPMN pós pré-processamento
    # =========
    if in_deg_unique(start_event) != 0:
        fail("StartEvent tem entradas (não representável no encoding).")
    if out_deg_unique(end_event) != 0:
        fail("EndEvent tem saídas (não representável no encoding).")

    # =========
    # 9) Checagem de conectividade Start -> End, e tasks no caminho (opcional)
    # =========
    def reachable_from(start) -> Set[object]:
        seen: Set[object] = set()
        q = deque([start])
        seen.add(start)
        while q:
            cur = q.popleft()
            for nxt in out_targets_sorted(cur):
                if nxt in seen:
                    continue
                seen.add(nxt)
                q.append(nxt)
        return seen

    def can_reach(target) -> Set[object]:
        seen: Set[object] = set()
        q = deque([target])
        seen.add(target)
        while q:
            cur = q.popleft()
            for prv in in_sources_sorted(cur):
                if prv in seen:
                    continue
                seen.add(prv)
                q.append(prv)
        return seen

    reach_start = reachable_from(start_event)
    if end_event not in reach_start:
        fail("EndEvent não é alcançável a partir do StartEvent.")

    if enforce_start_to_end_path:
        reach_end_rev = can_reach(end_event)
        for t in sorted([x for x in g.get_nodes() if is_task(x) and x.get_process() == process_id], key=node_key):
            if t not in reach_start:
                fail(f"Task não é alcançável a partir do StartEvent: {fmt_node(t)}")
            if t not in reach_end_rev:
                fail(f"Task não consegue alcançar o EndEvent: {fmt_node(t)}")

    # =========
    # 10) Caches pós-pré-processamento e travessias unificadas
    # =========
    def cache_key_node(node) -> str:
        nid = _get_id_str(node)
        if nid != "":
            return nid
        return f"obj_{id(node)}"

    _forward_cache: Dict[str, Tuple[Set[object], Set[int]]] = {}
    _backward_cache: Dict[str, Tuple[Set[object], Set[int]]] = {}

    def analyze_forward(src_node) -> Tuple[Set[object], Set[int]]:
        """
        Retorna:
          - succ_set: tasks/end imediatamente alcançáveis (colapsando gateways)
          - split_types: tipos de gateways de ramificação vistos (out_deg_unique>1) antes de atingir tasks/end
        """
        k = cache_key_node(src_node)
        if k in _forward_cache:
            return _forward_cache[k]

        succ_set: Set[object] = set()
        split_types: Set[int] = set()

        q = deque(out_targets_sorted(src_node))
        visited: Set[object] = set()

        while q:
            cur = q.popleft()
            if cur in visited:
                continue
            visited.add(cur)

            if not hasattr(cur, "get_process") or cur.get_process() != process_id:
                fail(f"Intermediário fora do processo principal no caminho (forward): {fmt_node(cur)}")

            if is_task(cur) or is_end(cur):
                succ_set.add(cur)
                continue

            if is_gateway(cur):
                # Melhoria portada: falhar cedo se ainda existir gateway misto.
                indeg_u = in_deg_unique(cur)
                outdeg_u = out_deg_unique(cur)
                if indeg_u > 1 and outdeg_u > 1:
                    fail(f"Gateway misto encontrado durante forward (deveria ter sido normalizado): {fmt_node(cur)}")

                if outdeg_u > 1:
                    split_types.add(gateway_type_value(cur))
                for nxt in out_targets_sorted(cur):
                    q.append(nxt)
                continue

            fail(f"Intermediário não suportado no caminho (forward): {fmt_node(cur)}")

        _forward_cache[k] = (succ_set, split_types)
        return succ_set, split_types

    def analyze_backward(dst_node) -> Tuple[Set[object], Set[int]]:
        """
        Retorna:
          - pred_set: tasks/start imediatamente alcançáveis no reverso (colapsando gateways)
          - join_types: tipos de gateways de convergência vistos (in_deg_unique>1) antes de atingir tasks/start
        """
        k = cache_key_node(dst_node)
        if k in _backward_cache:
            return _backward_cache[k]

        pred_set: Set[object] = set()
        join_types: Set[int] = set()

        q = deque(in_sources_sorted(dst_node))
        visited: Set[object] = set()

        while q:
            cur = q.popleft()
            if cur in visited:
                continue
            visited.add(cur)

            if not hasattr(cur, "get_process") or cur.get_process() != process_id:
                fail(f"Intermediário fora do processo principal no caminho (backward): {fmt_node(cur)}")

            if is_task(cur) or is_start(cur):
                pred_set.add(cur)
                continue

            if is_gateway(cur):
                # Melhoria portada: falhar cedo se ainda existir gateway misto.
                indeg_u = in_deg_unique(cur)
                outdeg_u = out_deg_unique(cur)
                if indeg_u > 1 and outdeg_u > 1:
                    fail(f"Gateway misto encontrado durante backward (deveria ter sido normalizado): {fmt_node(cur)}")

                if indeg_u > 1:
                    join_types.add(gateway_type_value(cur))
                for prv in in_sources_sorted(cur):
                    q.append(prv)
                continue

            fail(f"Intermediário não suportado no caminho (backward): {fmt_node(cur)}")

        _backward_cache[k] = (pred_set, join_types)
        return pred_set, join_types

    # =========
    # 11) Construir cromossomo (n+1 x n+1)
    # =========
    crom = np.zeros((n + 1, n + 1), dtype=np.int64)
    A = crom[:n, :n]
    meta_split = crom[:n, -1]
    meta_join = crom[-1, :n]

    # =========
    # 12) Marcar tasks ausentes no BPMN como desligadas (-1/-1)
    # =========
    present_label: Set[str] = set(name_to_task.keys()) | {"begin", "end"}

    for lab in alphabet_list:
        if lab in ("begin", "end"):
            continue
        if lab not in present_label:
            i = idx_of[lab]
            meta_split[i] = -1
            meta_join[i] = -1

    def node_for_label(lab: str):
        if lab == "begin":
            return start_event
        if lab == "end":
            return end_event
        return name_to_task.get(lab)

    # =========
    # 13) Popular A e meta_split
    # =========
    for src_lab in alphabet_list:
        if src_lab == "end":
            continue

        src_node = node_for_label(src_lab)
        if src_node is None:
            continue

        succ_set, split_types = analyze_forward(src_node)

        for s in succ_set:
            if is_end(s):
                tgt_lab = "end"
            else:
                tgt_lab = _norm_label(_get_name_str(s))

            if tgt_lab not in idx_of:
                sugestoes = difflib.get_close_matches(tgt_lab, list(idx_of.keys()), n=5, cutoff=0.82)
                extra = f" Sugestões: {sugestoes}" if sugestoes else ""
                fail(f"Sucessor {tgt_lab!r} não está no alphabet.{extra}")

            A[idx_of[src_lab], idx_of[tgt_lab]] = 1

        i = idx_of[src_lab]
        if src_lab not in present_label:
            continue

        if len(split_types) == 0:
            meta_split[i] = 0
        elif len(split_types) == 1:
            t = next(iter(split_types))
            if len(succ_set) <= 1:
                fail(
                    f"Split detectado a partir de '{src_lab}', mas há {len(succ_set)} sucessor(es) imediato(s). "
                    f"Split escondido não é representável no seu encoding."
                )
            meta_split[i] = int(t)
        else:
            fail(f"Cadeia de splits com tipos diferentes a partir de {fmt_node(src_node)}: {sorted(split_types)}")

    # =========
    # 14) Popular meta_join
    # =========
    for dst_lab in alphabet_list:
        if dst_lab == "begin":
            continue

        dst_node = node_for_label(dst_lab)
        if dst_node is None:
            continue

        pred_set, join_types = analyze_backward(dst_node)

        j = idx_of[dst_lab]
        if dst_lab not in present_label:
            continue

        if len(join_types) == 0:
            meta_join[j] = 0
        elif len(join_types) == 1:
            t = next(iter(join_types))
            if len(pred_set) <= 1:
                fail(
                    f"Join detectado antes de '{dst_lab}', mas há {len(pred_set)} predecessor(es) imediato(s). "
                    f"Join escondido não é representável no seu encoding."
                )
            meta_join[j] = int(t)
        else:
            fail(f"Cadeia de joins com tipos diferentes antes de {fmt_node(dst_node)}: {sorted(join_types)}")

    begin_idx = idx_of["begin"]
    end_idx = idx_of["end"]

    if int(A[:, begin_idx].sum()) != 0:
        fail("O modelo induzido gera arestas que chegam em 'begin' (StartEvent). Isso não é representável no seu encoding.")
    if int(A[end_idx, :].sum()) != 0:
        fail("O modelo induzido gera arestas saindo de 'end' (EndEvent). Isso não é representável no seu encoding.")

    A[:, begin_idx] = 0
    A[end_idx, :] = 0
    A[begin_idx, begin_idx] = 0
    A[end_idx, end_idx] = 0

    # =========
    # 15) Política de self-loop
    # =========
    if not allow_self_loops:
        for i in range(n):
            if i in (begin_idx, end_idx):
                continue
            if A[i, i] == 1:
                lab = alphabet_list[i]
                fail(f"Self-loop A[{i},{i}] em '{lab}', mas allow_self_loops=False.")

    # =========
    # 16) Checagens finais de representabilidade
    # =========
    outdeg = A.sum(axis=1)
    indeg = A.sum(axis=0)

    for lab in alphabet_list:
        i = idx_of[lab]
        if meta_split[i] == -1 and meta_join[i] == -1:
            continue
        if meta_split[i] > 0 and int(outdeg[i]) <= 1:
            fail(f"Split detectado em '{lab}', mas A tem outdeg<=1. Split escondido não representável no seu encoding.")
        if int(outdeg[i]) >= 2 and meta_split[i] == 0:
            fail(f"A tem outdeg>=2 em '{lab}', mas meta_split==0. Inconsistência.")

    for lab in alphabet_list:
        j = idx_of[lab]
        if meta_split[j] == -1 and meta_join[j] == -1:
            continue
        if meta_join[j] > 0 and int(indeg[j]) <= 1:
            fail(f"Join detectado em '{lab}', mas A tem indeg<=1. Join escondido não representável no seu encoding.")
        if int(indeg[j]) >= 2 and meta_join[j] == 0:
            fail(f"A tem indeg>=2 em '{lab}', mas meta_join==0. Inconsistência.")

    for lab in alphabet_list:
        k = idx_of[lab]
        if meta_split[k] == -1 and meta_join[k] == -1:
            A[k, :] = 0
            A[:, k] = 0

    return crom


def generate_bpmn_graph_from_cromossome(
    cromossome: np.ndarray,
    alphabet: Sequence[str],
    *,
    process_id: str = "Process_001",
    name: str = "",
    begin_label: str = "begin",
    end_label: str = "end",
    strict_shape: bool = True,
    allow_self_loops: bool = True,
    strict_meta: bool = True,
    fallback_gateway_type: int = 1,  # usado só se strict_meta=False (1=XOR, 2=AND, 3=OR)
    do_cleanup: bool = False,
    cleanup_max_iters: int = 50,
    enforce_start_to_end_path: bool = False,
    validate_gateway_chains: bool = False,
) -> BPMN:
    """
    Versão final consolidada:

    - Base best_hybrid: flags separadas (validate_gateway_chains e enforce_start_to_end_path), defaults adequados para GA.
    - Metadado (hybrid_v4): domínio {-1,0,1,2,3}, regra "-1 só como (-1,-1)" e:
        * nó ligado com out_deg<=1 => meta_split==0
        * nó ligado com in_deg<=1  => meta_join==0
        * nó ligado com out_deg>1  => meta_split in {1,2,3}
        * nó ligado com in_deg>1   => meta_join  in {1,2,3}
    - Canonização após cleanup (hybrid_v4):
        * StartEvent com 1 saída única
        * EndEvent com 1 entrada única
    - Guard rail: quando strict_shape=False, se begin/end existem no alphabet mas caem fora do recorte [:n], falha cedo.
    - Proíbe labels duplicados no alphabet (incluindo begin/end).
    """

    if not isinstance(cromossome, np.ndarray):
        cromossome = np.array(cromossome)

    alphabet_list = [str(x) for x in alphabet]
    n_alpha = len(alphabet_list)

    if n_alpha < 2:
        raise ChromosomeConversionError(cromossome, "[bpmn-gen-fail] Alphabet inválido, precisa ter pelo menos 2 labels.")
    if len(set(alphabet_list)) != len(alphabet_list):
        raise ChromosomeConversionError(cromossome, "[bpmn-gen-fail] alphabet contém labels duplicados, isso é proibido no seu encoding.")
    if cromossome.ndim != 2:
        raise ChromosomeConversionError(cromossome, f"[bpmn-gen-fail] cromossome precisa ser 2D, shape={getattr(cromossome, 'shape', None)}")

    pid = str(process_id)
    if pid.strip() == "":
        raise ChromosomeConversionError(cromossome, "[bpmn-gen-fail] process_id vazio.")

    # =========================
    # 0) Determinar n e recortar crom (n+1 x n+1)
    # =========================
    if strict_shape:
        expected = (n_alpha + 1, n_alpha + 1)
        if tuple(cromossome.shape) != expected:
            raise ChromosomeConversionError(cromossome, f"[bpmn-gen-fail] Unexpected cromossome shape {cromossome.shape}, expected {expected}")
        n = n_alpha
        crom = cromossome
    else:
        n_mat = min(int(cromossome.shape[0]) - 1, int(cromossome.shape[1]) - 1)
        n = min(n_alpha, n_mat)
        if n <= 0:
            raise ChromosomeConversionError(cromossome, f"[bpmn-gen-fail] Invalid n computed: n={n}, alphabet={n_alpha}, cromossome.shape={cromossome.shape}")

        A_full = cromossome[:-1, :-1]
        if A_full.shape[0] > n and np.any(A_full[n:, :n] != 0):
            raise ChromosomeConversionError(cromossome, "[bpmn-gen-fail] Non-zero adjacency abaixo de n, truncation iria descartar arestas.")
        if A_full.shape[1] > n and np.any(A_full[:n, n:] != 0):
            raise ChromosomeConversionError(cromossome, "[bpmn-gen-fail] Non-zero adjacency à direita de n, truncation iria descartar arestas.")

        if cromossome.shape[0] < n + 1 or cromossome.shape[1] < n + 1:
            raise ChromosomeConversionError(cromossome, f"[bpmn-gen-fail] cromossome shape {cromossome.shape} não comporta n+1={n+1}.")
        crom = cromossome[: n + 1, : n + 1]

        # Guard rail: begin/end não podem existir fora do recorte usado
        def _index_if_present(lst: List[str], label: str) -> Optional[int]:
            try:
                return lst.index(label)
            except ValueError:
                return None

        bi_full = _index_if_present(alphabet_list, begin_label)
        ei_full = _index_if_present(alphabet_list, end_label)
        if bi_full is not None and bi_full >= n:
            raise ChromosomeConversionError(cromossome, 
                f"[bpmn-gen-fail] begin_label='{begin_label}' existe no alphabet, mas está fora do recorte usado (idx={bi_full} >= n={n}). "
                f"Com strict_shape=False, isso indica entrada inconsistente."
            )
        if ei_full is not None and ei_full >= n:
            raise ChromosomeConversionError(cromossome, 
                f"[bpmn-gen-fail] end_label='{end_label}' existe no alphabet, mas está fora do recorte usado (idx={ei_full} >= n={n}). "
                f"Com strict_shape=False, isso indica entrada inconsistente."
            )

    A = crom[:n, :n].astype(np.int64, copy=False)
    meta_split = crom[:n, n].astype(np.int64, copy=False)
    meta_join = crom[n, :n].astype(np.int64, copy=False)

    def _is_meta_type(v: int) -> bool:
        return int(v) in (1, 2, 3)

    def _is_meta_domain(v: int) -> bool:
        return int(v) in (-1, 0, 1, 2, 3)

    # =========================
    # 1) begin/end por label (ou fallback)
    # =========================
    def find_idx(label: str) -> Optional[int]:
        try:
            return alphabet_list.index(label)
        except ValueError:
            return None

    begin_idx_raw = find_idx(begin_label)
    end_idx_raw = find_idx(end_label)
    begin_from_label = begin_idx_raw is not None
    end_from_label = end_idx_raw is not None

    begin_idx = begin_idx_raw
    end_idx = end_idx_raw

    enabled = [not (int(meta_split[i]) == -1 and int(meta_join[i]) == -1) for i in range(n)]

    if begin_idx is not None:
        begin_idx = int(begin_idx)
        if begin_idx >= n:
            raise ChromosomeConversionError(cromossome, f"[bpmn-gen-fail] begin_label='{begin_label}' encontrado no alphabet, mas idx={begin_idx} está fora do recorte n={n}.")
        if not enabled[begin_idx]:
            raise ChromosomeConversionError(cromossome, f"[bpmn-gen-fail] '{begin_label}' está desligado (-1,-1). Isso viola o encoding.")

    if end_idx is not None:
        end_idx = int(end_idx)
        if end_idx >= n:
            raise ChromosomeConversionError(cromossome, f"[bpmn-gen-fail] end_label='{end_label}' encontrado no alphabet, mas idx={end_idx} está fora do recorte n={n}.")
        if not enabled[end_idx]:
            raise ChromosomeConversionError(cromossome, f"[bpmn-gen-fail] '{end_label}' está desligado (-1,-1). Isso viola o encoding.")

    if begin_idx is None or end_idx is None:
        present_idxs = [i for i in range(n) if enabled[i]]
        if len(present_idxs) < 2:
            raise ChromosomeConversionError(cromossome, "[bpmn-gen-fail] Não há nós suficientes (presentes) para inferir begin/end.")
        if begin_idx is None:
            begin_idx = present_idxs[0]
        if end_idx is None:
            end_idx = present_idxs[-1]

    begin_idx = int(begin_idx)
    end_idx = int(end_idx)
    if begin_idx == end_idx:
        raise ChromosomeConversionError(cromossome, "[bpmn-gen-fail] begin_idx == end_idx, não é representável como StartEvent e EndEvent.")

    enabled[begin_idx] = True
    enabled[end_idx] = True

    # =========================
    # 2) A2, zera nós ausentes
    # =========================
    A2 = A.copy()
    for i in range(n):
        if not enabled[i]:
            A2[i, :] = 0
            A2[:, i] = 0

    # Restrições do encoding (begin/end)
    if int(A2[:, begin_idx].sum()) != 0:
        raise ChromosomeConversionError(cromossome, "[bpmn-gen-fail] Há arestas chegando em 'begin' (StartEvent). Não representável no encoding.")
    if int(A2[end_idx, :].sum()) != 0:
        raise ChromosomeConversionError(cromossome, "[bpmn-gen-fail] Há arestas saindo de 'end' (EndEvent). Não representável no encoding.")

    if int(A2[begin_idx, begin_idx]) != 0:
        raise ChromosomeConversionError(cromossome, "[bpmn-gen-fail] Self-loop em 'begin' não é permitido.")
    if int(A2[end_idx, end_idx]) != 0:
        raise ChromosomeConversionError(cromossome, "[bpmn-gen-fail] Self-loop em 'end' não é permitido.")

    if not allow_self_loops:
        for i in range(n):
            if not enabled[i] or i in (begin_idx, end_idx):
                continue
            if int(A2[i, i]) == 1:
                raise ChromosomeConversionError(cromossome, f"[bpmn-gen-fail] Self-loop em '{alphabet_list[i]}', mas allow_self_loops=False.")

    out_deg = A2.sum(axis=1).astype(int)
    in_deg = A2.sum(axis=0).astype(int)

    # =========================
    # 2.1) Domínio e consistência metadado (rigor final)
    # =========================
    if strict_meta:
        for i in range(n):
            ms = int(meta_split[i])
            mj = int(meta_join[i])

            if not _is_meta_domain(ms) or not _is_meta_domain(mj):
                raise ChromosomeConversionError(cromossome, f"[bpmn-gen-fail] Metadado fora do domínio em '{alphabet_list[i]}': meta_split={ms}, meta_join={mj}.")

            # -1 só pode aparecer como (-1,-1)
            if (ms == -1) != (mj == -1):
                raise ChromosomeConversionError(cromossome, 
                    f"[bpmn-gen-fail] Metadado inválido em '{alphabet_list[i]}': meta_split={ms}, meta_join={mj} "
                    f"(no encoding, -1 só pode aparecer como (-1,-1))."
                )

            node_enabled = enabled[i]  # enabled deriva de (-1,-1)
            if not node_enabled:
                if not (ms == -1 and mj == -1):
                    raise ChromosomeConversionError(cromossome, f"[bpmn-gen-fail] Nó desligado '{alphabet_list[i]}' com metadado inválido: meta_split={ms}, meta_join={mj}.")
                continue

            # nó ligado: não pode ter -1
            if ms == -1 or mj == -1:
                raise ChromosomeConversionError(cromossome, f"[bpmn-gen-fail] Nó ligado '{alphabet_list[i]}' com metadado inválido: meta_split={ms}, meta_join={mj}.")

            # Regras específicas para begin/end apenas se vieram de labels reais
            if i == begin_idx and begin_from_label and mj != 0:
                raise ChromosomeConversionError(cromossome, f"[bpmn-gen-fail] begin '{begin_label}' com meta_join={mj}, esperado 0.")
            if i == end_idx and end_from_label and ms != 0:
                raise ChromosomeConversionError(cromossome, f"[bpmn-gen-fail] end '{end_label}' com meta_split={ms}, esperado 0.")

            od = int(out_deg[i])
            idg = int(in_deg[i])

            if od > 1:
                if ms not in (1, 2, 3):
                    raise ChromosomeConversionError(cromossome, f"[bpmn-gen-fail] outdeg>1 em '{alphabet_list[i]}', mas meta_split={ms}.")
            else:
                if ms != 0:
                    raise ChromosomeConversionError(cromossome, f"[bpmn-gen-fail] meta_split={ms} em '{alphabet_list[i]}', mas outdeg<=1.")

            if idg > 1:
                if mj not in (1, 2, 3):
                    raise ChromosomeConversionError(cromossome, f"[bpmn-gen-fail] indeg>1 em '{alphabet_list[i]}', mas meta_join={mj}.")
            else:
                if mj != 0:
                    raise ChromosomeConversionError(cromossome, f"[bpmn-gen-fail] meta_join={mj} em '{alphabet_list[i]}', mas indeg<=1.")

    # =========================
    # 3) Criar BPMN, StartEvent, EndEvent
    # =========================
    bpmn = BPMN(process_id=pid, name=name)

    start = BPMN.StartEvent(id="StartEvent_1", name="", process=pid)
    end = BPMN.EndEvent(id="EndEvent_1", name="", process=pid)
    bpmn.add_node(start)
    bpmn.add_node(end)

    # =========================
    # 4) Criar tasks
    # =========================
    nodes: Dict[int, object] = {begin_idx: start, end_idx: end}

    for i in range(n):
        if not enabled[i] or i in (begin_idx, end_idx):
            continue

        lab = str(alphabet_list[i]).strip()
        if lab == "":
            raise ChromosomeConversionError(cromossome, f"[bpmn-gen-fail] Task com label vazio no índice {i}.")
        if lab in (begin_label, end_label):
            raise ChromosomeConversionError(cromossome, f"[bpmn-gen-fail] Task com label reservado '{lab}' fora de begin/end. Isso quebra bpmn_to_cromossome.")

        t = BPMN.Task(id=f"Activity_{i}", name=lab, process=pid)
        bpmn.add_node(t)
        nodes[i] = t

    # =========================
    # 5) Gateways por grau
    # =========================
    def _gw_type_or_fallback(v: int) -> int:
        v = int(v)
        if _is_meta_type(v):
            return v
        return int(fallback_gateway_type)

    def _make_gateway(gw_type: int, gw_id: str, *, direction) -> object:
        if gw_type == 2:
            return BPMN.ParallelGateway(id=gw_id, name="", gateway_direction=direction, process=pid)
        if gw_type == 3:
            return BPMN.InclusiveGateway(id=gw_id, name="", gateway_direction=direction, process=pid)
        return BPMN.ExclusiveGateway(id=gw_id, name="", gateway_direction=direction, process=pid)

    split_gw: Dict[int, object] = {}
    join_gw: Dict[int, object] = {}

    for i in range(n):
        if not enabled[i] or i == end_idx:
            continue
        if int(out_deg[i]) > 1:
            v = int(meta_split[i])
            if strict_meta and v not in (1, 2, 3):
                raise ChromosomeConversionError(cromossome, f"[bpmn-gen-fail] outdeg>1 em '{alphabet_list[i]}', mas meta_split={v}.")
            gw_type = v if v in (1, 2, 3) else _gw_type_or_fallback(v)
            gw = _make_gateway(gw_type, f"Gateway_split_{i}", direction=BPMN.Gateway.Direction.DIVERGING)
            bpmn.add_node(gw)
            split_gw[i] = gw

    for j in range(n):
        if not enabled[j] or j == begin_idx:
            continue
        if int(in_deg[j]) > 1:
            v = int(meta_join[j])
            if strict_meta and v not in (1, 2, 3):
                raise ChromosomeConversionError(cromossome, f"[bpmn-gen-fail] indeg>1 em '{alphabet_list[j]}', mas meta_join={v}.")
            gw_type = v if v in (1, 2, 3) else _gw_type_or_fallback(v)
            gw = _make_gateway(gw_type, f"Gateway_join_{j}", direction=BPMN.Gateway.Direction.CONVERGING)
            bpmn.add_node(gw)
            join_gw[j] = gw

    # =========================
    # 6) Flows (dedup por existência real no grafo)
    # =========================
    used_flow_ids: Set[str] = set()

    def _refresh_used_flow_ids() -> None:
        used_flow_ids.clear()
        for f in list(bpmn.get_flows()):
            fid = f.get_id()
            if fid is not None:
                used_flow_ids.add(str(fid))

    _refresh_used_flow_ids()

    def _unique_flow_id(base: str) -> str:
        if base not in used_flow_ids:
            used_flow_ids.add(base)
            return base
        k = 1
        cand = f"{base}_{k}"
        while cand in used_flow_ids:
            k += 1
            cand = f"{base}_{k}"
        used_flow_ids.add(cand)
        return cand

    def _flow_exists(src, tgt) -> bool:
        for f in list(src.get_out_arcs()):
            if f.get_target() == tgt:
                return True
        return False

    def _remove_flow(f) -> None:
        fid = f.get_id()
        if f in bpmn.get_flows():
            bpmn.remove_flow(f)
        if fid is not None:
            used_flow_ids.discard(str(fid))

    def add_flow_if_absent(src_node, tgt_node, base_id: str) -> None:
        if _flow_exists(src_node, tgt_node):
            return
        fid = _unique_flow_id(base_id)
        bpmn.add_flow(BPMN.SequenceFlow(src_node, tgt_node, id=fid, name="", process=pid))

    # join -> nó
    for j, gw in join_gw.items():
        add_flow_if_absent(gw, nodes[j], f"Flow_join_{j}_to_{nodes[j].get_id()}")

    # nó -> split
    for i, gw in split_gw.items():
        add_flow_if_absent(nodes[i], gw, f"Flow_{nodes[i].get_id()}_to_split_{i}")

    # adjacência
    for i in range(n):
        if not enabled[i] or i == end_idx:
            continue
        for j in range(n):
            if not enabled[j] or j == begin_idx:
                continue
            if int(A2[i, j]) != 1:
                continue
            src_eff = split_gw.get(i, nodes[i])
            tgt_eff = join_gw.get(j, nodes[j])
            add_flow_if_absent(src_eff, tgt_eff, f"Flow_{src_eff.get_id()}_to_{tgt_eff.get_id()}")

    # sanidade mínima
    if len(list(start.get_in_arcs())) != 0:
        raise ChromosomeConversionError(cromossome, "[bpmn-gen-fail] StartEvent ficou com entradas. Inconsistente.")
    if len(list(end.get_out_arcs())) != 0:
        raise ChromosomeConversionError(cromossome, "[bpmn-gen-fail] EndEvent ficou com saídas. Inconsistente.")
    if len(list(start.get_out_arcs())) == 0:
        raise ChromosomeConversionError(cromossome, "[bpmn-gen-fail] StartEvent ficou sem saídas.")
    if len(list(end.get_in_arcs())) == 0:
        raise ChromosomeConversionError(cromossome, "[bpmn-gen-fail] EndEvent ficou sem entradas.")

    # =========================
    # Helpers de processo e grau único
    # =========================
    def _is_gateway(node) -> bool:
        return isinstance(node, (BPMN.ExclusiveGateway, BPMN.ParallelGateway, BPMN.InclusiveGateway))

    def _nodes_in_process() -> List[object]:
        return [x for x in bpmn.get_nodes() if hasattr(x, "get_process") and x.get_process() == pid]

    def _unique_out_targets(node) -> List[object]:
        seen: Set[object] = set()
        out: List[object] = []
        for f in list(node.get_out_arcs()):
            tgt = f.get_target()
            if tgt in seen:
                continue
            seen.add(tgt)
            out.append(tgt)
        return out

    def _unique_in_sources(node) -> List[object]:
        seen: Set[object] = set()
        out: List[object] = []
        for f in list(node.get_in_arcs()):
            src = f.get_source()
            if src in seen:
                continue
            seen.add(src)
            out.append(src)
        return out

    def _out_deg_unique(node) -> int:
        return len(_unique_out_targets(node))

    def _in_deg_unique(node) -> int:
        return len(_unique_in_sources(node))

    # =========================
    # 7) Cleanup opcional + canonização Start/End
    # =========================
    if do_cleanup:
        def _remove_node_with_incident_flows(node) -> None:
            for f in list(node.get_in_arcs()):
                _remove_flow(f)
            for f in list(node.get_out_arcs()):
                _remove_flow(f)
            if node in bpmn.get_nodes():
                bpmn.remove_node(node)

        def _dedup_parallel_flows_once() -> bool:
            removed = 0
            for src in _nodes_in_process():
                seen_tgts: Set[object] = set()
                for f in list(src.get_out_arcs()):
                    tgt = f.get_target()
                    if tgt in seen_tgts:
                        _remove_flow(f)
                        removed += 1
                    else:
                        seen_tgts.add(tgt)
            return removed > 0

        def _remove_orphan_gateways_once() -> bool:
            gws = [x for x in _nodes_in_process() if _is_gateway(x)]
            for gw in sorted(gws, key=lambda z: str(z.get_id())):
                if _in_deg_unique(gw) == 0 or _out_deg_unique(gw) == 0:
                    _remove_node_with_incident_flows(gw)
                    return True
            return False

        def _remove_degenerate_gateways_once() -> bool:
            gws = [x for x in _nodes_in_process() if _is_gateway(x)]
            for gw in sorted(gws, key=lambda z: str(z.get_id())):
                ins = _unique_in_sources(gw)
                outs = _unique_out_targets(gw)
                if len(ins) == 1 and len(outs) == 1:
                    pred = ins[0]
                    succ = outs[0]

                    if pred == succ:
                        if pred in (start, end):
                            raise ChromosomeConversionError(cromossome, "[bpmn-gen-fail] Self-loop induzido em StartEvent/EndEvent via gateway degenerado, inválido.")
                        if not allow_self_loops:
                            raise ChromosomeConversionError(cromossome, 
                                f"[bpmn-gen-fail] Estrutura induz self-loop via gateway degenerado (pred==succ) em gw id={gw.get_id()}, "
                                f"mas allow_self_loops=False."
                            )
                        add_flow_if_absent(pred, succ, f"Flow_cleanup_selfloop_{gw.get_id()}")
                        _remove_node_with_incident_flows(gw)
                        return True

                    add_flow_if_absent(pred, succ, f"Flow_cleanup_bypass_{gw.get_id()}")
                    _remove_node_with_incident_flows(gw)
                    return True
            return False

        guard = 0
        changed = True
        while changed:
            guard += 1
            if guard > int(cleanup_max_iters):
                raise ChromosomeConversionError(cromossome, "[bpmn-gen-fail] Cleanup excedeu limite de iterações, possível ciclo de reescrita.")
            changed = False
            changed |= _dedup_parallel_flows_once()
            changed |= _remove_orphan_gateways_once()
            changed |= _remove_degenerate_gateways_once()
            changed |= _dedup_parallel_flows_once()

        if len(list(start.get_in_arcs())) != 0:
            raise ChromosomeConversionError(cromossome, "[bpmn-gen-fail] StartEvent ficou com entradas após cleanup.")
        if len(list(end.get_out_arcs())) != 0:
            raise ChromosomeConversionError(cromossome, "[bpmn-gen-fail] EndEvent ficou com saídas após cleanup.")

        # Canonização Start/End
        if len(_unique_out_targets(start)) != 1:
            raise ChromosomeConversionError(cromossome, "[bpmn-gen-fail] StartEvent ficou com número de saídas diferente de 1 após cleanup.")
        if len(_unique_in_sources(end)) != 1:
            raise ChromosomeConversionError(cromossome, "[bpmn-gen-fail] EndEvent ficou com número de entradas diferente de 1 após cleanup.")

    # =========================
    # 8) Validação de cadeias de gateways (toggle separado)
    # =========================
    if validate_gateway_chains:
        def _gateway_code(node) -> int:
            if isinstance(node, BPMN.ExclusiveGateway):
                return 1
            if isinstance(node, BPMN.ParallelGateway):
                return 2
            if isinstance(node, BPMN.InclusiveGateway):
                return 3
            return 0

        def _walk_chain_forward(first_gw) -> int:
            t0 = _gateway_code(first_gw)
            cur = first_gw
            visited_ids: Set[str] = set()
            while True:
                cid = str(cur.get_id())
                if cid in visited_ids:
                    break
                visited_ids.add(cid)

                if not _is_gateway(cur):
                    break

                tcur = _gateway_code(cur)
                if tcur != 0 and t0 != 0 and tcur != t0:
                    raise ChromosomeConversionError(cromossome, 
                        f"[bpmn-gen-fail] Cadeia de gateways com tipos diferentes (forward) a partir de id='{first_gw.get_id()}': "
                        f"[{t0}, {tcur}]"
                    )

                if _in_deg_unique(cur) != 1 or _out_deg_unique(cur) != 1:
                    break

                nxt = list(cur.get_out_arcs())[0].get_target()
                if not _is_gateway(nxt):
                    break
                cur = nxt
            return t0

        def _walk_chain_backward(first_gw) -> int:
            t0 = _gateway_code(first_gw)
            cur = first_gw
            visited_ids: Set[str] = set()
            while True:
                cid = str(cur.get_id())
                if cid in visited_ids:
                    break
                visited_ids.add(cid)

                if not _is_gateway(cur):
                    break

                tcur = _gateway_code(cur)
                if tcur != 0 and t0 != 0 and tcur != t0:
                    raise ChromosomeConversionError(cromossome, 
                        f"[bpmn-gen-fail] Cadeia de gateways com tipos diferentes (backward) a partir de id='{first_gw.get_id()}': "
                        f"[{t0}, {tcur}]"
                    )

                if _in_deg_unique(cur) != 1 or _out_deg_unique(cur) != 1:
                    break

                prv = list(cur.get_in_arcs())[0].get_source()
                if not _is_gateway(prv):
                    break
                cur = prv
            return t0

        # splits
        for src in _nodes_in_process():
            if _out_deg_unique(src) <= 1:
                continue
            types: List[int] = []
            for f in list(src.get_out_arcs()):
                tgt = f.get_target()
                if _is_gateway(tgt):
                    types.append(_walk_chain_forward(tgt))
            if len(set(types)) > 1:
                raise ChromosomeConversionError(cromossome, 
                    f"[bpmn-gen-fail] Cadeia de splits com tipos diferentes a partir de type={type(src).__name__} "
                    f"id='{src.get_id()}' name='{src.get_name()}': {sorted(set(types))}"
                )

        # joins
        for tgt in _nodes_in_process():
            if _in_deg_unique(tgt) <= 1:
                continue
            types2: List[int] = []
            for f in list(tgt.get_in_arcs()):
                src = f.get_source()
                if _is_gateway(src):
                    types2.append(_walk_chain_backward(src))
            if len(set(types2)) > 1:
                raise ChromosomeConversionError(cromossome, 
                    f"[bpmn-gen-fail] Cadeia de joins com tipos diferentes a originar type={type(tgt).__name__} "
                    f"id='{tgt.get_id()}' name='{tgt.get_name()}': {sorted(set(types2))}"
                )

    # =========================
    # 9) Conectividade Start -> End (toggle separado)
    # =========================
    if enforce_start_to_end_path:
        def reachable_from(s) -> Set[object]:
            seen: Set[object] = set()
            q = deque([s])
            seen.add(s)
            while q:
                cur = q.popleft()
                for f in list(cur.get_out_arcs()):
                    nxt = f.get_target()
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    q.append(nxt)
            return seen

        def can_reach(t) -> Set[object]:
            seen: Set[object] = set()
            q = deque([t])
            seen.add(t)
            while q:
                cur = q.popleft()
                for f in list(cur.get_in_arcs()):
                    prv = f.get_source()
                    if prv in seen:
                        continue
                    seen.add(prv)
                    q.append(prv)
            return seen

        r1 = reachable_from(start)
        if end not in r1:
            raise ChromosomeConversionError(cromossome, "[bpmn-gen-fail] EndEvent não é alcançável a partir do StartEvent.")
        r2 = can_reach(end)

        for i in range(n):
            if not enabled[i] or i in (begin_idx, end_idx):
                continue
            node = nodes[i]
            if node not in r1:
                raise ChromosomeConversionError(cromossome, f"[bpmn-gen-fail] Task '{alphabet_list[i]}' não é alcançável a partir do StartEvent.")
            if node not in r2:
                raise ChromosomeConversionError(cromossome, f"[bpmn-gen-fail] Task '{alphabet_list[i]}' não consegue alcançar o EndEvent.")

    return bpmn
