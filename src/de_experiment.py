
import libsumo as traci
from datetime import datetime
from src import config
from src import road_config
from src.de_optimizer import differential_evolution
from src.simulation_libs import reload_sumo_with_state, apply_plan
from src.utils import get_green_split, log_cycle_result
from src.traffic_metrics import cycle_metrics, score_function

def seed_de_simulation(total_cycles=300):
    # Start SUMO
    sumoCmd = [
        config.SUMO_BINARY,
        "-c", config.SUMO_CFG,
        "--seed", str(config.SEED),
    ]
    log_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = ""

    elite = [0.5, config.CYCLE_LENGTH_DEFAULT]  # initial elite

    traci.start(sumoCmd)
    for cycle in range(total_cycles):
        # load default state 
        if(cycle == 0):
            traci.simulation.loadState(config.SUMO_FIRST_STATE)
        time_pased = traci.simulation.getTime()
        print(f"\nTime passed at cycle {cycle+1}: {time_pased}s.")
        # break on end
        if time_pased >= config.END_TIME:
            print(f"❌ Elapsed time exceed at: cycle {cycle+1}, stopping early.")
            break
        print(f"=== Cycle {cycle+1} (t={time_pased:.1f}s) ===")

        #  Save the current live SUMO state once
        traci.simulation.saveState(config.SUMO_STATE)

        #  Run DE using that snapshot as baseline
        result = differential_evolution(elite_last=elite, time_budget_s=20)
        s, C, O1, O2, score_best, elapsed_s, gen_history = (
            result.s, result.C, result.O1, result.O2,
            result.score, result.elapsed, result.gen_history
        )
        # elite = (s, C)

        #  Restore pre-optimization state before applying best plan
        reload_sumo_with_state(config.SUMO_STATE, config.sumo_cmd)
          
        #  Apply only the *best plan* to the live SUMO world
        g_main, g_cross = get_green_split(s, C)
        apply_plan(road_config.TL_ID, g_main, g_cross)

        # Evaluate performance under fixed plan
        O1, O2 = cycle_metrics(road_config.NS_LANES, road_config.EW_LANES, int(C))
        score_best = score_function(O1, O2)
            
        #  Log and write results
        print(
            f"Chosen split={s:.2f}, C={C:.1f}, "
            f"O1={O1:.3f}, O2={O2:.3f}, score={score_best:.3f}, time={elapsed_s:.2f}s, gen={len(gen_history)}"
        )
        # Save logs
        log_file = log_cycle_result(
            cycle + 1, s, C, O1, O2, score_best, elapsed_s,
            suffix=log_suffix, out_dir="logs", prefix="traffic_DE"
        )

    traci.close()    
    print(f"\n✅ Simulation completed — results saved to {log_file}")

def baseline_simulation(fixed_s=0.5, fixed_C=90, total_cycles=300):
    """
    Baseline simulation using a fixed signal plan (no optimization).
    Uses identical SUMO setup and logging as the DE version
    for direct comparison.
    """
    sumoCmd = [
        config.SUMO_BINARY,
        "-c", config.SUMO_CFG,
        "--seed", str(config.SEED),
    ]
    log_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = ""

    print(f"Running baseline simulation: s={fixed_s}, C={fixed_C}s for {total_cycles} cycles")

    traci.start(sumoCmd) 
    for cycle in range(total_cycles):
        # load default state 
        if(cycle == 0):
            traci.simulation.loadState(config.SUMO_FIRST_STATE)
        time_pased = traci.simulation.getTime()
        print(f"\nTime passed at cycle {cycle+1}: {time_pased}s.")
        if time_pased >= config.END_TIME:
            print(f"❌ No more vehicles at cycle {cycle+1}, stopping early.")
            break
        print(f"=== Cycle {cycle+1} (t={time_pased:.1f}s) ===")
        
        print("Vehicles currently in simulation:", traci.vehicle.getIDCount())
        # Apply fixed plan
        g_main, g_cross = get_green_split(fixed_s, fixed_C)
        apply_plan(road_config.TL_ID, g_main, g_cross)

        # Evaluate performance under fixed plan
        O1, O2 = cycle_metrics(road_config.NS_LANES, road_config.EW_LANES, int(fixed_C))
        score_best = score_function(O1, O2)
        elapsed_s = 0.0  # not optimized, so no runtime cost

        # Log
        print(
            f"Fixed split={fixed_s:.2f}, C={fixed_C:.1f}, "
            f"O1={O1:.3f}, O2={O2:.3f}, score={score_best:.3f}"
        )

        # Write results
        log_file = log_cycle_result(
            cycle + 1,
            fixed_s,
            fixed_C,
            O1,
            O2,
            score_best,
            elapsed_s,
            suffix=log_suffix,
            out_dir="logs",
            prefix="traffic_baseline",
        )
    traci.close()   
    print(f"\n✅ Baseline completed — results saved to {log_file}")


