
import libsumo as traci
from datetime import datetime
from src import config
from src import road_config
from src.de_optimizer import differential_evolution
from src.logging_ml_teacher import append_sample, extract_temporal_features, init_dataset_file
from src.simulation_libs import reload_sumo_with_state, apply_plan
from src.train_de_mlp import load_mlp_controller, mlp_predict_plan
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


def seed_de_teacher_simulation(total_cycles=300):
    # Start SUMO
    sumoCmd = [
        config.SUMO_BINARY,
        "-c", config.SUMO_CFG,
        "--seed", str(config.SEED),
    ]
    log_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = ""

    elite = [0.5, config.CYCLE_LENGTH_DEFAULT]  # initial elite
     # Start SUMO and load initial state once
    traci.start(sumoCmd)
    traci.simulation.loadState(config.SUMO_FIRST_STATE)

    # === Init dataset + temporal state ===
    init_dataset_file()

    # previous decision (for cycle 0, some default)
    prev_split = elite[0]
    prev_cycle = elite[1]

    # temporal trend initialization from the initial state
    q_ns_prev, q_ew_prev = 0.0, 0.0
    q_ns_ema = 0.0
    q_ew_ema = 0.0

    for cycle in range(total_cycles):
        time_pased = traci.simulation.getTime()
        print(f"\nTime passed at cycle {cycle+1}: {time_pased}s.")

        # break on end
        if time_pased >= config.END_TIME:
            print(f"❌ Elapsed time exceed at: cycle {cycle+1}, stopping early.")
            break

        print(f"=== Cycle {cycle+1} (t={time_pased:.1f}s) ===")

        # Temporal features at decision time (based only on past cycles)
        temporal_feats = extract_temporal_features(
            q_ns_prev,
            q_ew_prev,
            q_ns_ema,
            q_ew_ema,
            ema_alpha=config.EMA_ALPHA,
        )
        # update EMA state for next cycle
        q_ns_ema = temporal_feats["q_NS_ema"]
        q_ew_ema = temporal_feats["q_EW_ema"]

        # Save the current live SUMO state once (Sim A snapshot → Sim B base)
        traci.simulation.saveState(config.SUMO_STATE)

        # Run DE using that snapshot as baseline (Sim B)
        result = differential_evolution(elite_last=elite)
        s, C, O1, O2, score_best, elapsed_s, gen_history = (
            result.s, result.C, result.O1, result.O2,
            result.score, result.elapsed, result.gen_history
        )
        elite = (s, C)

        # Log (inputs at decision time, labels from DE) BEFORE advancing Sim A
        row = [
            cycle + 1,
            # temporal trend
            temporal_feats["q_NS_ema"],
            temporal_feats["q_EW_ema"],
            # previous cycle aggregates (from Sim A)
            temporal_feats["q_NS_prev"],  # = ns_delay_prev
            temporal_feats["q_EW_prev"],  # = ew_delay_prev
            # previous decision
            prev_split,
            prev_cycle,
            # DE teacher labels (best plan for this cycle)
            s,
            C,
        ]
        append_sample(row)

        #  Restore pre-optimization state before applying best plan
        reload_sumo_with_state(config.SUMO_STATE, config.sumo_cmd)
          
        #  Apply only the *best plan* to the live SUMO world
        g_main, g_cross = get_green_split(s, C)
        apply_plan(road_config.TL_ID, g_main, g_cross)

        # Evaluate performance under fixed plan
        O1, O2, ns_delay_curr, ew_delay_curr = cycle_metrics(road_config.NS_LANES, road_config.EW_LANES, int(C), is_return_delay_proxies=True)
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

         # 9) Update state for next cycle
        prev_split = s
        prev_cycle = C

        # these are what the NEXT cycle will see as "previous delays"
        q_ns_prev = ns_delay_curr   # ns_delay_prev for next cycle
        q_ew_prev = ew_delay_curr   # ew_delay_prev for next cycle

    traci.close()    
    print(f"\n✅ Simulation completed — results saved to {log_file}")


def seed_mlp_controller_simulation(total_cycles=300):
    # 0) Load trained MLP controller
    model, x_scaler, y_scaler, feature_cols, target_cols, device = load_mlp_controller()

    # 1) Start SUMO
    sumoCmd = [
        config.SUMO_BINARY,
        "-c", config.SUMO_CFG,
        "--seed", str(config.SEED),
    ]
    log_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = ""

    traci.start(sumoCmd)
    traci.simulation.loadState(config.SUMO_FIRST_STATE)

    # --- initial state ---
    prev_split = 0.5
    prev_cycle = config.CYCLE_LENGTH_DEFAULT

    q_ns_prev, q_ew_prev = 0.0, 0.0
    q_ns_ema = 0.0
    q_ew_ema = 0.0

    for cycle in range(total_cycles):
        time_passed = traci.simulation.getTime()
        print(f"\nTime passed at cycle {cycle+1}: {time_passed}s.")

        if time_passed >= config.END_TIME:
            print(f"Elapsed time exceed at: cycle {cycle+1}, stopping early.")
            break

        print(f"=== MLP Cycle {cycle+1} (t={time_passed:.1f}s) ===")

        # Temporal features at decision time (based on past cycles only)
        temporal_feats = extract_temporal_features(
            q_ns_prev,
            q_ew_prev,
            q_ns_ema,
            q_ew_ema,
            ema_alpha=config.EMA_ALPHA,
        )
        q_ns_ema = temporal_feats["q_NS_ema"]
        q_ew_ema = temporal_feats["q_EW_ema"]

        # Save state if you still want the option to compare with DE later
        traci.simulation.saveState(config.SUMO_STATE)

        # --- MLP decides best (s, C) for this cycle ---
        s, C = mlp_predict_plan(
            model,
            x_scaler,
            y_scaler,
            temporal_feats=temporal_feats,
            prev_split=prev_split,
            prev_cycle=prev_cycle,
            device=device,
        )

        # Apply to SUMO
        g_main, g_cross = get_green_split(s, C)
        apply_plan(road_config.TL_ID, g_main, g_cross)

        # Evaluate under fixed MLP plan
        O1, O2, ns_delay_curr, ew_delay_curr = cycle_metrics(
            road_config.NS_LANES,
            road_config.EW_LANES,
            int(C),
            is_return_delay_proxies=True,
        )
        score_best = score_function(O1, O2)

        print(
            f"[MLP] split={s:.3f}, C={C:.1f}, "
            f"O1={O1:.3f}, O2={O2:.3f}, score={score_best:.3f}"
        )

        # Optional: log results similar to DE
        log_file = log_cycle_result(
            cycle + 1, s, C, O1, O2, score_best, elapsed_s=0.0,
            suffix=log_suffix, out_dir="logs", prefix="traffic_MLP"
        )

        # update state for next cycle
        prev_split = s
        prev_cycle = C
        q_ns_prev = ns_delay_curr
        q_ew_prev = ew_delay_curr

    traci.close()
    print(f"\n✅ MLP simulation completed — results saved to {log_file}")