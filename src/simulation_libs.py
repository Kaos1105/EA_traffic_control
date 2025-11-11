import libsumo as traci
from src import config
from src import road_config
#  cmd = [
#         "-c", SUMO_CFG,
#         "--seed", str(SEED),
#         "--load-state", state_path,
#         "--begin", str(begin_time)
#     ]

def reload_sumo_with_state(state_path, sumo_cmd):
    """
    Reload the current SUMO simulation from a saved state safely.
    (Wrapper around traci.load)
    Args:
        sumo_cfg (str): Path to the SUMO configuration file (.sumocfg).
        seed (int): Random seed for reproducibility.
        state_path (str): Path to the saved simulation state (.xml or .xml.gz).
        begin_time (float): Simulation time to resume from (matches saveState time).
    """
    # --- Parse snapshot time from XML header ---
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            for line in f:
                if 'snapshot' in line and 'time=' in line:
                    # Extract value like time="2791.00"
                    time_str = line.split('time="')[1].split('"')[0]
                    begin_time = float(time_str)
                    break
            else:
                raise ValueError("No <snapshot> header with 'time' found in state file.")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to read time from state file '{state_path}': {e}")
    
   
    traci.load([*sumo_cmd, "--load-state", state_path,
        "--begin", str(begin_time)])
    
def find_avg_halt_range(ns_lanes, ew_lanes):
    ns_delay_proxy = ew_delay_proxy = 0
    max_avg_halt = 0
    min_avg_halt = 99999

    min_fairness = 1
    max_fairness = 0
    while traci.simulation.getTime() < config.END_TIME:
        ns_delay_proxy = 0
        ew_delay_proxy = 0
        for _ in range(config.CYCLE_LENGTH_DEFAULT):
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
                
            # Total average queue length
            total_avg_halt = (ns_delay_proxy + ew_delay_proxy)
             # Fairness (bounded 0–1)
            fainess = abs(ns_delay_proxy - ew_delay_proxy) / (ns_delay_proxy + ew_delay_proxy + 1e-5)
        if(total_avg_halt > max_avg_halt):
            max_avg_halt = total_avg_halt
        if(total_avg_halt < min_avg_halt):
            min_avg_halt = total_avg_halt
        if(fainess > max_fairness):
            max_fairness = fainess
        if(fainess < min_fairness):
            min_fairness = fainess

    print(f"❌ Stopping at END_TIME.")
    usable_time = config.CYCLE_LENGTH_DEFAULT - config.LOST_TIME
    
    return max_avg_halt/usable_time,min_avg_halt/usable_time,max_fairness,min_fairness    

def apply_plan(tl_id, g_main, g_cross, amber=3, all_red=3):
    """
    Define a 2-phase signal plan (NS and EW) with amber and all-red times.
    """
    amrber_red_phase_duration = amber + all_red
    phases = [
        traci.trafficlight.Phase(g_main, "rrrrrGGggGGG"),   # NS green
        traci.trafficlight.Phase(amrber_red_phase_duration, "rrrrryyyyyyy"),    # NS amber
        traci.trafficlight.Phase(g_cross, "GGGGGrrrrrrr"),  # EW green
        traci.trafficlight.Phase(amrber_red_phase_duration, "yyyyyrrrrrrr"),    # EW amber
    ]

    logic = traci.trafficlight.Logic("custom_logic", 0, 0, phases)
    traci.trafficlight.setCompleteRedYellowGreenDefinition(tl_id, logic)