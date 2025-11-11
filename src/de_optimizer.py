import random
import numpy as np
import time
from src import config
from src.traffic_metrics import score_function, robust_evaluate, evaluate_candidate, DEResult

 # Include previous elite as first individual (temporal continuity)
def init_population(bounds, pop_size, elite_last=None):
    pop = [np.array([random.uniform(*b) for b in bounds]) for _ in range(pop_size)]
    if elite_last is not None:
        pop[0] = np.array([elite_last[0], elite_last[1]])  # temporal continuity
    return pop

# Early stopping check using std
def early_stop_check(gen_history, patience, min_delta, elapsed, time_budget_s):
    if len(gen_history) > patience:
        recent_scores = [h[-1] for h in gen_history[-patience:]]
        score_std = np.std(recent_scores)
        if score_std < min_delta:
            print(f"Early stopping: score std {score_std:.6f} < {min_delta}")
            return True
        
    if elapsed >= time_budget_s:
            print(f"Early stopping: time lapsed {elapsed}s")
            return True
    return False

 # --- Select top-K and re-evaluate robustly
def final_robust_selection(pop, scores, K_ratio=0.1):
    K = round(len(pop) * K_ratio)
    ranked = sorted(
        [(pop[i], scores[i], score_function(*scores[i])) for i in range(len(pop))],
        key=lambda x: x[2]
    )
    topK = ranked[:K]
    evaluated = [(x, *robust_evaluate(x)) for x, _, _ in topK] #(x, robust_score, mean_O1, mean_O2)
    winner, best_robust_score, O1_first, O2_first = min(evaluated, key=lambda z: z[1])
    s_best, C_best = winner
    return s_best, C_best, O1_first, O2_first, best_robust_score
    
def reflect_in_bounds(x, lo, hi):
    if x < lo: return lo + (lo - x)
    if x > hi: return hi - (x - hi)
    return x
    
def evolve_generation(pop, scores, bounds, F, CR):
    scale = np.array([1.0, 1.0 / (bounds[1][1] - bounds[1][0])])  # scale by range
    F_vec = F*scale
    pop_size = len(pop)
    new_pop, new_scores = pop.copy(), scores.copy()
    dimensions = len(bounds); lower_bound, upper_bound = np.asarray(bounds).T

    for i in range(pop_size):
        idxs = [idx for idx in range(pop_size) if idx != i]
        choice_idx = np.random.choice(idxs, 3, replace=False)
        r1, r2, r3 = pop[choice_idx[0]], pop[choice_idx[1]], pop[choice_idx[2]]

         # --- Mutation ---
        mutant = r1 +  F_vec * (r2 - r3)
        # --- Reflection bounds ---
        mutant[0] = reflect_in_bounds(mutant[0], lower_bound[0], upper_bound[0])
        mutant[1] = reflect_in_bounds(mutant[1], lower_bound[1], upper_bound[1])
        
        # Crossover (ensure at least one mutant gene)
        cross_points = np.random.rand(dimensions) < CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, dimensions)] = True
        trial = np.where(cross_points, mutant, pop[i])    

        # Evaluate and selection
        O1_t, O2_t = evaluate_candidate(trial)
        O1_i, O2_i = scores[i]
        score_t = score_function(O1_t, O2_t)
        score_i = score_function(O1_i, O2_i)

        if score_t < score_i:
            new_pop[i], new_scores[i] = trial, (O1_t, O2_t)

    return new_pop, new_scores

def differential_evolution(
    elite_last=None,
    time_budget_s=60,
    pop_size=12,
    F=0.5,
    CR=0.8,
    patience=5,
    min_delta=1e-4,
):
    """
    Differential Evolution (DE/rand/1/bin)
    for 2 parameters: (s, C)
    Optimizes two objectives (O1, O2) within a strict time budget.

    Returns: DEResult(s_best, C_best, O1_best, O2_best, score_best, elapsed)
    """
    start = time.perf_counter()
    bounds = [(config.MIN_GREEN_SPLIT, 1-config.MIN_GREEN_SPLIT), (config.MIN_CYCLE_LENGTH, config.MAX_CYCLE_LENGTH)]  # (s_min, s_max), (C_min, C_max)

    pop = init_population(bounds, pop_size, elite_last)
    scores = [None] * pop_size

    # --- Evaluate initial population
    for i in range(pop_size):
        O1, O2 = evaluate_candidate(pop[i])
        scores[i] = (O1, O2)

    gen = 0
    gen_history = []  # store per-generation bests
    # --- DE optimization loop (time-bounded)
    while time.perf_counter() - start < time_budget_s:
        gen += 1
        pop, scores = evolve_generation(pop, scores, bounds, F, CR)

        # Summary
        best_idx = min(range(pop_size),
                       key=lambda k: score_function(scores[k][0], scores[k][1]))
        best = pop[best_idx]
        s_best, C_best = best
        O1_best, O2_best = scores[best_idx]
        score_best = score_function(O1_best, O2_best)
        elapsed = time.perf_counter() - start

        # print(f"gen {gen:02d} | t={elapsed:4.1f}s | "
        #       f"s={s_best:.3f} C={C_best:.1f} "
        #       f"O1={O1_best:.3f} O2={O2_best:.3f} score={score_best:.3f}")

        gen_history.append((elapsed, s_best, C_best, O1_best, O2_best, score_best))

        if early_stop_check(gen_history, patience, min_delta, elapsed, time_budget_s):
            break

    #TODO: skip multi cycle evaluation
    # --- Final robust evaluation
    # s_final, C_final, O1_first_eval, O2_first_eval, best_robust_score = final_robust_selection(pop, scores)
    
    # # Re-evaluate the winner once for single-cycle score
    # score_final = score_function(O1_first_eval, O2_first_eval)

    # elapsed = time.perf_counter() - start

    # return DEResult(s_final, C_final, O1_first_eval, O2_first_eval, score_final, best_robust_score, elapsed, gen_history)

    elapsed = time.perf_counter() - start

    return DEResult(s_best, C_best, O1_best, O2_best, score_best, elapsed, gen_history)
