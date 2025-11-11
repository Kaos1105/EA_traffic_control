SEED = 8
SUMO_GUI_BINARY  = "sumo-gui"  # or "sumo-gui"
SUMO_BINARY  = "sumo"  # or "sumo-gui"
SUMO_CFG  = "./sumo_cfg/simulation.sumocfg"  # your .sumocfg file
SUMO_STATE = "./sumo_cfg/simulation_state.xml"  # your initial state file
SUMO_FIRST_STATE = "./sumo_cfg/simulation_first_state.xml"  # your initial state file
LOG_DIR= "logs"

# Global timing constants
REDUCE_AVG_WAIT_TIME_W, FAIRNESS_W = 0.7, 0.3 # prioritize reduce wait time
CYCLE_LENGTH_DEFAULT = 90   # s
MAX_CYCLE_LENGTH = 120      # s
MIN_CYCLE_LENGTH = 60      # s
LOST_TIME = 12              # s (amber + all-red total)
GREEN_MIN = 15             # s per direction
MIN_GREEN_SPLIT = GREEN_MIN / (MIN_CYCLE_LENGTH-LOST_TIME)  # minimum green split ratio
END_TIME = 16200
CHECK_POINT_INTERVAL = 1800

#Delay rate (queue area per usable second)
MIN_DELAY_RATE_PER_CYCLE = 0
MAX_DELAY_RATE_PER_CYCLE = 40

sumo_cmd = [
        "-c", SUMO_CFG,
        "--seed", str(SEED),
        ]