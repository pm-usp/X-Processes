import fitness as fit
import numpy as np
import operators as opr
from numba import njit
import os
import glob
import uuid
import time
import subprocess
import pm4py
from pathlib import Path
import bpmn as bp
import petrinets as pn
import re
import unicodedata
import multiprocessing as mp
import traceback
import datetime as dt
import pytz

SHOW_SEED_ATTEMPTS = 0
SHOW_SEED_FAILS = 0

SPLITMINER_TIMEOUT_S = 120
IM_TIMEOUT_S = 120

def _norm_label(v) -> str:
    s = unicodedata.normalize("NFKC", str(v))
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def process_trace(trace):
    return set(trace)


def create_alphabet(listOfLists_log):
    alphabet_set = set()
    norm_to_raw = {}
    for trace in listOfLists_log:
        for ev in trace:
            lab = _norm_label(ev)
            if lab == "":
                continue
            prev = norm_to_raw.get(lab)
            raw = str(ev)
            if prev is not None and prev != raw:
                raise ValueError(f"Colisão de label após normalização: {repr(prev)} e {repr(raw)} viraram {repr(lab)}")
            norm_to_raw[lab] = raw
            alphabet_set.add(lab)
    alphabet = np.array(sorted(alphabet_set), dtype=object)
    alphabet = np.insert(alphabet, 0, "begin")
    alphabet = np.append(alphabet, "end")
    print("Number of activities:", len(alphabet) - 2, flush=True)
    return alphabet


def process_entry(entry):
    entry.insert(0, 'begin')
    entry.append('end')
    return entry


def process_log(listOfLists_log):
    listOfLists_log = [process_entry(entry) for entry in listOfLists_log]
    return listOfLists_log


def initialize_population(population_size, alphabet, reference_cromossome, fitness_weight, precision_weight, generalization_weight, simplicity_weight, simplicity_criterion_methods, input_log_name, input_xes_log_path, round, island, generation, cache_fitness, cache_soundness, cache_petri_net, cache_n_tokens, act_fix_mode, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log, filetimestamp):
    population = np.zeros((population_size, (len(alphabet) + 1), (len(alphabet)) + 1), dtype=np.int64)
    reference_cromossome = np.array(reference_cromossome, dtype=np.int64)
    for i in range(population_size):
        population[i] = create_initial_individual(population[i], alphabet, reference_cromossome, cache_n_tokens, act_fix_mode)
        sound = pn.is_sound(population[i], alphabet, cache_soundness, cache_petri_net)
        while (not sound):
            population[i] = initialize_individual(len(alphabet) + 1)
            population[i] = create_initial_individual(population[i], alphabet, reference_cromossome, cache_n_tokens, act_fix_mode)
            sound = pn.is_sound(population[i], alphabet, cache_soundness, cache_petri_net)
    population = seed_population_for_island(population, island, reference_cromossome, input_xes_log_path, alphabet, cache_soundness, cache_petri_net, act_fix_mode, simplicity_criterion_methods)
    evaluated_population, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log = fit.evaluate_population(population, alphabet, fitness_weight, precision_weight, generalization_weight, simplicity_weight, simplicity_criterion_methods, input_log_name, cache_fitness, cache_petri_net, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log, filetimestamp)
    return population, evaluated_population, reference_cromossome, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log


def seed_covers_all_activities_if_fixed(candidate, alphabet, act_fix_mode) -> bool:
    if not act_fix_mode:
        return True
    total_of_tasks = len(alphabet) - 2
    candidate = np.array(candidate, dtype=np.int64, copy=False)
    for t in range(1, total_of_tasks + 1):
        if candidate[t, -1] == -1 or candidate[-1, t] == -1:
            return False
        if not opr.is_there_at_least_one_row_active_task(candidate, t):
            return False
        if not opr.is_there_at_least_one_column_active_task(candidate, t):
            return False
    return True


def seed_population_for_island(population, island: int, reference_cromossome, input_xes_log_path: str, alphabet, cache_soundness, cache_petri_net, act_fix_mode, simplicity_criterion_methods):
    seed_plan = {
        0: [(0, "ref", None), (1, "sm", "0.05"), (2, "im", 0.20)],
        1: [(0, "sm", "0.00"), (1, "sm", "0.50"), (2, "sm", "1.00")],
        2: [(0, "im", 0.00), (1, "im", 0.50), (2, "im", 1.00)],
        3: [(0, "sm", "0.25"), (1, "sm", "0.75")],
        4: [(0, "im", 0.75), (1, "im", 0.25)],
    }

    def build_candidate(kind: str, param, simplicity_criterion_methods, full_xes_log):
        if kind == "ref":
            temp_cromossome = reference_cromossome.copy()
            cromossome_with_gateways = create_gateways(temp_cromossome)
            return cromossome_with_gateways
        if kind == "sm":
            return discover_model_with_sm(input_xes_log_path, alphabet, simplicity_criterion_methods, epsilon=param)
        if kind == "im":
            return discover_model_with_im_timeout(input_xes_log_path=input_xes_log_path, alphabet=alphabet, noise_threshold=float(param), timeout_s=IM_TIMEOUT_S, full_xes_log=full_xes_log)
        raise ValueError(f"Seed kind inválido: {kind}")

    needs_im = any(kind == "im" for _, kind, _ in seed_plan.get(island, []))
    if needs_im and os.name != "nt":
        xes_abs = os.path.abspath(input_xes_log_path)
        if not os.path.isfile(xes_abs):
            raise FileNotFoundError(f"XES não encontrado: {xes_abs}")
        full_xes_log = pm4py.read_xes(xes_abs)
    else:
        full_xes_log = None

    for pop_idx, kind, param in seed_plan.get(island, []):
        if pop_idx >= len(population):
            continue
        if SHOW_SEED_ATTEMPTS:
            print(f"[seed] ilha={island} pop_idx={pop_idx} kind={kind} param={param}", flush=True)
        try:
            candidate = build_candidate(kind, param, simplicity_criterion_methods, full_xes_log)
            candidate = np.array(candidate, dtype=np.int64, copy=False)
            if not seed_covers_all_activities_if_fixed(candidate, alphabet, act_fix_mode):
                if SHOW_SEED_FAILS:
                    print(
                        f"[seed-fail] ilha={island} pop_idx={pop_idx} kind={kind} param={param} "
                        f"act_fix_mode=1 seed_sem_cobertura_total_de_atividades",
                        flush=True,
                    )
                continue
        except bp.ChromosomeConversionError as e:
            if SHOW_SEED_FAILS:
                print(
                    f"[seed-fail] ilha={island} pop_idx={pop_idx} kind={kind} param={param} "
                    f"{type(e).__module__}.{type(e).__name__}: {e}",
                    flush=True,
                )
            continue
        except Exception as e:
            if SHOW_SEED_FAILS:
                print(
                    f"[seed-fail] ilha={island} pop_idx={pop_idx} kind={kind} param={param} "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
            continue
        if pn.is_sound(candidate, alphabet, cache_soundness, cache_petri_net):
            population[pop_idx] = candidate
        else:
            if SHOW_SEED_FAILS:
                print(f"[seed-fail] ilha={island} pop_idx={pop_idx} kind={kind} param={param} nao-sound", flush=True)
    return population


def _im_worker(mode: str, xes_path: str, full_xes_log, noise_threshold: float, alphabet, q):
    try:
        if mode == "path":
            log_obj = pm4py.read_xes(xes_path)
        else:
            log_obj = full_xes_log

        net, im, fm = pm4py.discover_petri_net_inductive(log_obj, noise_threshold=noise_threshold)
        bpmn_model = pm4py.convert_to_bpmn(net, im, fm)
        crom = bp.bpmn_to_cromossome(bpmn_model, alphabet, mutate_input=True)

        q.put(("ok", np.array(crom, dtype=np.int64)))
    except Exception:
        q.put(("err", traceback.format_exc()))


class SeedTimeoutError(TimeoutError):
    pass


def discover_model_with_im_timeout(input_xes_log_path: str, alphabet, noise_threshold: float, timeout_s: float, full_xes_log=None):
    # Windows: spawn, não passe full_xes_log (ficaria caro por pickle)
    # Linux: use fork e passe full_xes_log para não reler o XES
    if os.name == "nt":
        ctx = mp.get_context("spawn")
        mode = "path"
        xes_path = str(Path(input_xes_log_path).expanduser().resolve())
        log_arg = None
    else:
        ctx = mp.get_context("fork")
        mode = "log"
        xes_path = ""
        log_arg = full_xes_log
    q = ctx.Queue(maxsize=1)
    p = ctx.Process(target=_im_worker, args=(mode, xes_path, log_arg, float(noise_threshold), alphabet, q))
    p.start()
    p.join(float(timeout_s))
    if p.is_alive():
        p.terminate()
        p.join()
        raise SeedTimeoutError(f"IM timeout após {float(timeout_s):.3f}s (noise_threshold={noise_threshold})")
    if q.empty():
        raise RuntimeError("IM terminou sem retorno no Queue (provável falha/crash).")
    status, payload = q.get()
    if status == "err":
        raise RuntimeError(payload)
    return payload  # cromossome (np.int64)


def discover_model_with_im(full_xes_log, alphabet, simplicity_criterion_methods, noise_threshold: float):
    petrinet, im, fm = pm4py.discover_petri_net_inductive(full_xes_log, noise_threshold=noise_threshold)
    bpmn = pm4py.convert_to_bpmn(petrinet, im, fm)
    cromossome = bp.bpmn_to_cromossome(bpmn, alphabet, mutate_input=True)
    return cromossome


def get_splitminer_home() -> Path:
    env = os.getenv("SPLITMINER2_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent / "java"


def discover_model_with_sm(input_xes_log_path: str, alphabet, simplicity_criterion_methods, epsilon: str = "0.05"):
    bpmn = splitminer_java(input_xes_log_path, epsilon=epsilon)
    cromossome = bp.bpmn_to_cromossome(bpmn, alphabet, mutate_input=True)
    return cromossome


def splitminer_java(input_xes_log_path: str, epsilon: str = "0.05", timeout_s: float = SPLITMINER_TIMEOUT_S):
    sm2_dir = get_splitminer_home()
    xes_abs = Path(input_xes_log_path).expanduser().resolve()
    if not xes_abs.is_file():
        raise FileNotFoundError(f"XES não encontrado: {xes_abs}")
    out_prefix = f"sm2_out_{uuid.uuid4().hex}"
    expected_bpmn = sm2_dir / f"{out_prefix}.bpmn"
    t0 = time.time()
    try:
        try:
            res = run_splitminer(str(xes_abs), out_prefix, epsilon=epsilon, timeout_s=timeout_s)
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout or ""
            stderr = e.stderr or ""
            raise RuntimeError(
                "SplitMiner estourou o timeout.\n"
                f"TIMEOUT: {float(timeout_s):.3f}s\n"
                f"CMD: {e.cmd}\n"
                f"STDOUT:\n{stdout}\n"
                f"STDERR:\n{stderr}\n"
            ) from e
        stdout = res.stdout or ""
        stderr = res.stderr or ""
        erro_textual = (
            ("ERROR" in stdout.upper())
            or ("EXCEPTION" in stderr.upper())
            or ("NUMBERFORMATEXCEPTION" in stderr.upper())
        )
        if res.returncode != 0 or erro_textual:
            raise RuntimeError(
                "SplitMiner falhou.\n"
                f"RET: {res.returncode}\n"
                f"CMD: {res.args}\n"
                f"STDOUT:\n{stdout}\n"
                f"STDERR:\n{stderr}\n"
            )
        if (not expected_bpmn.is_file()) or (expected_bpmn.stat().st_size == 0):
            recent = []
            for p in glob.glob(str(sm2_dir / "*.bpmn")):
                try:
                    if os.path.getmtime(p) >= t0 - 2:
                        recent.append(p)
                except OSError:
                    pass
            raise RuntimeError(
                "SplitMiner retornou sem erro, mas o BPMN esperado não apareceu.\n"
                f"Esperado: {expected_bpmn}\n"
                f"BPMNs recentes em {sm2_dir}:\n" + "\n".join(recent) + "\n"
                f"STDOUT:\n{stdout}\n"
                f"STDERR:\n{stderr}\n"
            )
        return pm4py.read_bpmn(str(expected_bpmn))
    finally:
        if expected_bpmn.exists():
            expected_bpmn.unlink()


def run_splitminer(xes_path: str, out_model_prefix: str, epsilon: str = "0.05", timeout_s: float = SPLITMINER_TIMEOUT_S,) -> subprocess.CompletedProcess:
    epsilon = str(epsilon)
    sm2_dir = get_splitminer_home()
    if not sm2_dir.is_dir():
        raise FileNotFoundError(f"Pasta do SplitMiner não é diretório: {sm2_dir}")
    jar_abs = sm2_dir / "sm2.jar"
    lib_dir = sm2_dir / "lib"
    if not jar_abs.is_file():
        raise FileNotFoundError(f"Não achei sm2.jar em: {jar_abs}")
    if not lib_dir.is_dir():
        raise FileNotFoundError(f"Não achei lib/ em: {lib_dir}")
    class_path = f"{str(jar_abs)}{os.pathsep}{str(lib_dir / '*')}"
    cmd = [
        "java",
        "-cp",
        class_path,
        "au.edu.unimelb.services.ServiceProvider",
        "SM2",
        xes_path,
        out_model_prefix,
        epsilon,
    ]
    return subprocess.run(cmd, cwd=str(sm2_dir), capture_output=True, text=True, timeout=float(timeout_s) if timeout_s is not None else None)


@njit(cache=True)
def initialize_individual(number_of_tasks):
    return np.zeros((number_of_tasks, number_of_tasks), dtype=np.int64)


@njit(cache=True)
def create_empty_individual_task(number_of_tasks):
    return [0] * (number_of_tasks + 1)


def create_DFG(listOfLists_log, alphabet):
    idx = {str(alphabet[i]): i for i in range(len(alphabet))}
    DFG = [create_empty_individual_task(len(alphabet)) for _ in range(len(alphabet) + 1)]
    for trace in listOfLists_log:
        for k in range(len(trace) - 1):
            current_task = idx[trace[k]]
            next_task = idx[trace[k + 1]]
            DFG[current_task][next_task] = 1
    return DFG


def get_task_id(task, alphabet):
    return np.where(alphabet == task)[0][0]


def create_initial_individual(cromossome, alphabet, reference_cromossome, cache_n_tokens, act_fix_mode):
    total_of_tasks = len(alphabet) - 2
    alpha = 0.75  # quanto maior, aumenta a probabilidade de iniciar com um número maior de atividades off
    beta = 2.5    # quanto menor, aumenta a probabilidade de iniciar com um número maior de atividades off
    alpha_beta_factor = np.random.beta(alpha, beta)
    cromossome = create_tasks(total_of_tasks, reference_cromossome, cromossome, act_fix_mode, alpha_beta_factor)
    cromossome = create_gateways(cromossome)
    cromossome, number_of_produced_single_tokens, number_of_consumed_single_tokens, number_of_produced_XOR_tokens, number_of_consumed_XOR_tokens, number_of_produced_AND_tokens, number_of_consumed_AND_tokens, number_of_produced_OR_tokens, number_of_consumed_OR_tokens = opr.count_number_of_tokens(cromossome, cache_n_tokens)
    cromossome = increase_or_decrease_tokens(cromossome, number_of_produced_single_tokens, number_of_consumed_single_tokens, number_of_produced_XOR_tokens, number_of_consumed_XOR_tokens, number_of_produced_AND_tokens, number_of_consumed_AND_tokens, number_of_produced_OR_tokens, number_of_consumed_OR_tokens)
    return cromossome


@njit(cache=True)
def sample_1_to_n_without_replacement(n, k):
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k > n:
        k = n
    pool = np.empty(n, dtype=np.int64)
    for i in range(n):
        pool[i] = i + 1  # 1..n
    for i in range(k):
        j = i + np.random.randint(0, n - i)
        tmp = pool[i]
        pool[i] = pool[j]
        pool[j] = tmp
    return pool[:k].copy()


@njit(cache=True)
def create_tasks(total_of_tasks, reference_cromossome, cromossome, act_fix_mode, alpha_beta_factor):
    num_of_start_off_tasks = 0
    if not act_fix_mode:
        num_of_start_off_tasks = int((alpha_beta_factor) * (total_of_tasks - 1))
    list_of_start_off_tasks = sample_1_to_n_without_replacement(total_of_tasks, num_of_start_off_tasks)
    for task in list_of_start_off_tasks:
        cromossome[task, -1] = -1
        cromossome[-1, task] = -1
    valid_start_tasks = [i for i in range(1, total_of_tasks + 1) if reference_cromossome[0][i] == 1 and i not in list_of_start_off_tasks]
    valid_start_tasks = np.array(valid_start_tasks, dtype=np.int64)
    if valid_start_tasks.size == 0:
        valid_start_tasks = np.array([i for i in range(1, total_of_tasks + 1) if i not in list_of_start_off_tasks], dtype=np.int64)
    for _ in range(np.random.randint(1, int((total_of_tasks + len(list_of_start_off_tasks) + 3) * 0.5))):
        x1 = np.random.choice(valid_start_tasks)
        cromossome[0, x1] = 1
        count = 0
        while True:
            valid_next_tasks = [i for i in range(1, total_of_tasks + 2) if reference_cromossome[x1][i] == 1 and i not in list_of_start_off_tasks]
            valid_next_tasks = np.array(valid_next_tasks, dtype=np.int64)
            if valid_next_tasks.size == 0:
                valid_next_tasks = np.array([i for i in range(1, total_of_tasks + 2) if i not in list_of_start_off_tasks], dtype=np.int64)
            x2 = np.random.choice(valid_next_tasks)
            cromossome[x1][x2] = 1
            x1 = x2
            if x2 == total_of_tasks + 1:
                break
            count += 1
            if count == total_of_tasks + 2:
                break
    for n in range(len(cromossome) - 2):
        if not opr.is_there_at_least_one_row_active_task(cromossome, n) and cromossome[n, -1] != -1:
            valid_tasks = [i for i in range(1, len(cromossome) - 1) if reference_cromossome[n][i] == 1 and i not in list_of_start_off_tasks]
            valid_tasks = np.array(valid_tasks, dtype=np.int64)
            if valid_tasks.size == 0:
                valid_tasks = np.array([i for i in range(1, len(cromossome) - 1) if i not in list_of_start_off_tasks], dtype=np.int64)
            cromossome[n][np.random.choice(valid_tasks)] = 1
    for n in range(1, len(cromossome) - 1):
        if not opr.is_there_at_least_one_column_active_task(cromossome, n) and cromossome[-1, n] != -1:
            valid_tasks = [i for i in range(0, len(cromossome) - 2) if reference_cromossome[i][n] == 1 and i not in list_of_start_off_tasks]
            valid_tasks = np.array(valid_tasks, dtype=np.int64)
            if valid_tasks.size == 0:
                valid_tasks = np.array([i for i in range(0, len(cromossome) - 2) if i not in list_of_start_off_tasks], dtype=np.int64)
            cromossome[np.random.choice(valid_tasks)][n] = 1
    return cromossome


@njit(cache=True)
def create_gateways(cromossome):
    for n in range(len(cromossome) - 2):
        if opr.is_there_at_least_two_row_active_tasks(cromossome, n) and cromossome[n, -1] != -1:
            cromossome[n][-1] = 1
    for n in range(1, len(cromossome) - 1):
        if opr.is_there_at_least_two_column_active_tasks(cromossome, n) and cromossome[-1, n] != -1:
            cromossome[-1][n] = 1
    return cromossome


@njit(cache=True)
def increase_or_decrease_tokens(cromossome, number_of_produced_single_tokens, number_of_consumed_single_tokens, number_of_produced_XOR_tokens, number_of_consumed_XOR_tokens, number_of_produced_AND_tokens, number_of_consumed_AND_tokens, number_of_produced_OR_tokens, number_of_consumed_OR_tokens):
    if (number_of_produced_single_tokens + number_of_produced_XOR_tokens + number_of_produced_AND_tokens + number_of_produced_OR_tokens) > (number_of_consumed_single_tokens + number_of_consumed_XOR_tokens + number_of_consumed_AND_tokens + number_of_consumed_OR_tokens):
        opr.increase_number_of_consumed_tokens(cromossome, number_of_produced_single_tokens, number_of_produced_XOR_tokens, number_of_produced_AND_tokens, number_of_produced_OR_tokens)
    elif (number_of_produced_single_tokens + number_of_produced_XOR_tokens + number_of_produced_AND_tokens + number_of_produced_OR_tokens) < (number_of_consumed_single_tokens + number_of_consumed_XOR_tokens + number_of_consumed_AND_tokens + number_of_consumed_OR_tokens):
        opr.increase_number_of_produced_tokens(cromossome, number_of_consumed_single_tokens, number_of_consumed_XOR_tokens, number_of_consumed_AND_tokens, number_of_consumed_OR_tokens)
    return cromossome