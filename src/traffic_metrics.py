import libsumo as traci
import numpy as np
from src import config
from src import road_config
from src.simulation_libs import reload_sumo_with_state, apply_plan
from src.utils import get_green_split
from dataclasses import dataclass

@dataclass
class DEResult:
    s: float
    C: float
    O1: float
    O2: float
    score: float
    elapsed: float
    gen_history: list

def cycle_metrics(ns_lanes, ew_lanes, steps):
    ns_delay_proxy = ew_delay_proxy = 0

    for _ in range(steps):
        traci.simulationStep()
        signal_state = traci.trafficlight.getRedYellowGreenState(road_config.TL_ID)

         # Split signal state: first 5 (EW), last 7 (NS)
        ew_state = signal_state[:5]
        ns_state = signal_state[5:]

        # Detect if direction is in red
        ew_red = "r" in ew_state
        ns_red = "r" in ns_state
        # Accumulate waiting times and vehicles only when direction is red
        if ns_red:
            ns_delay_proxy += sum(traci.lane.getLastStepHaltingNumber(l) for l in ns_lanes)
        if ew_red:
            ew_delay_proxy += sum(traci.lane.getLastStepHaltingNumber(l) for l in ew_lanes)
            
    # Fairness (bounded 0–1)
    O2_norm = abs(ns_delay_proxy - ew_delay_proxy) / (ns_delay_proxy + ew_delay_proxy + 1e-5)
    # Total average queue length
    delay_rate = (ns_delay_proxy + ew_delay_proxy)/(steps-config.LOST_TIME)
    O1_norm = (delay_rate - config.MIN_DELAY_RATE_PER_CYCLE) / (config.MAX_DELAY_RATE_PER_CYCLE - config.MIN_DELAY_RATE_PER_CYCLE)

    return O1_norm, O2_norm

# Evaluation function
def evaluate(s, C, isReset=True):
    try:    
        # Restore identical starting conditions before evaluating this candidate
        if isReset:
            reload_sumo_with_state(config.SUMO_STATE, config.sumo_cmd)
        g_main, g_cross = get_green_split(s, C)
        apply_plan(road_config.TL_ID, g_main, g_cross)
        steps = int(C)
        O1_norm, O2_norm = cycle_metrics(road_config.NS_LANES, road_config.EW_LANES, steps)

    except traci.exceptions.FatalTraCIError as e:
        print("⚠️ SUMO crashed during evaluation:", e)
        return C, 1  # return a default
    return O1_norm, O2_norm

def score_function(O1, O2):
    score = (config.REDUCE_AVG_WAIT_TIME_W * O1) + (config.FAIRNESS_W * O2)
    return score

# x consist of split-Cycle
def evaluate_candidate(x):
    s, C = float(x[0]), round(float(x[1]))  # round C for realism
    O1, O2 = evaluate(s, C)
    return O1, O2

def robust_evaluate(x, n_cycles=2, λ=0.2):
    """Returns (robust_score, mean_O1, mean_O2)"""
    s, C = float(x[0]), round(float(x[1]))
    # load previous state before evaluate robust
    reload_sumo_with_state(config.SUMO_STATE, config.sumo_cmd)
    scores = []
    O1s = []
    O2s = []
    for _ in range(n_cycles):
        O1, O2 = evaluate(s, C, isReset=False)  # continue simulation
        O1s.append(O1)
        O2s.append(O2)
        scores.append(score_function(O1, O2))
    robust_score = np.mean(scores) + λ * np.std(scores)
    # return O1, O2 of first evaluation
    return robust_score, O1s[0], O2s[0]