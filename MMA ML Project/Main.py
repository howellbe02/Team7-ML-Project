import os
import re
import logging
import pandas as pd
from datetime import datetime
from MarkovModel import states, markov_transition_matrix, simulate_chain, determine_winner
from AccuracyProbs import (
    get_binomial_probs,
    get_poisson_probs,
    get_ground_head_shot_prob,
    get_standing_head_shot_prob,
    get_gc_probs
)
# File paths
base_path = "./Skill Estimates/Skill Estimates Pre-2023-05-05"
match_path = "./data/NewMMAMatches.xlsx"
predictions_dir = "./predictions"
os.makedirs(predictions_dir, exist_ok=True)

# Determine next file number
existing_files = os.listdir(predictions_dir)
def next_number(prefix, ext):
    pattern = re.compile(rf"{re.escape(prefix)}_(\d+){re.escape(ext)}")
    nums = [int(m.group(1)) for f in existing_files if (m := pattern.fullmatch(f))]
    return max(nums, default=0) + 1

run_id = next_number("markov_log", ".txt")
log_file = os.path.join(predictions_dir, f"markov_log_{run_id}.txt")
excel_file = os.path.join(predictions_dir, f"fight_predictions_{run_id}.xlsx")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# Simulation settings
N_CHAINS = 2500
cutoff_date = pd.Timestamp("2023-05-05")

# Load matches
matches = pd.read_excel(match_path)
matches["Date"] = pd.to_datetime(matches["Date"], errors="coerce")
matches = matches[matches["Date"] > cutoff_date]
matches = matches.dropna(subset=["Fighter i ID", "Fighter j ID", "Cleaned Weight Class"])
matches["Fighter i ID"] = matches["Fighter i ID"].astype(int).astype(str)
matches["Fighter j ID"] = matches["Fighter j ID"].astype(int).astype(str)

results = []
seen_matchups = set()

# Process fights by weight class
for weight_class, group in matches.groupby("Cleaned Weight Class"):
    class_path = os.path.join(base_path, weight_class)
    if not os.path.isdir(class_path):
        continue

    weight_results = []

    try:
        logging.info(f"\n=== {weight_class} ===")

        # Collect matchups and unique fighter IDs
        matchups = list(zip(group["Fighter i ID"], group["Fighter j ID"]))
        fighter_ids = sorted(set(group["Fighter i ID"]).union(set(group["Fighter j ID"])))

        # Load skill estimates specific to matchups
        binom = get_binomial_probs(class_path, matchups)
        poisson = get_poisson_probs(class_path, matchups)
        ghp = get_ground_head_shot_prob(class_path, fighter_ids)
        shp = get_standing_head_shot_prob(class_path, fighter_ids)
        gc_probs = get_gc_probs(class_path, matchups)

        for fighter_i, fighter_j in matchups:
            pair = tuple(sorted([fighter_i, fighter_j]))
            if pair in seen_matchups:
                continue
            seen_matchups.add(pair)

            try:
                matrix = markov_transition_matrix(
                    fighter_i, fighter_j,
                    binomial_probs=binom,
                    poisson_probs=poisson,
                    gamma_probs=gc_probs,
                    ghp_probs=ghp,
                    shp_probs=shp
                )

                wins_i = wins_j = 0
                for _ in range(N_CHAINS):
                    chain, metrics = simulate_chain(matrix, states)
                    result = determine_winner(chain, metrics)
                    if result == "fighter_i": wins_i += 1
                    elif result == "fighter_j": wins_j += 1

                if fighter_i < fighter_j:
                    outcome = {
                        "Weight Class": weight_class,
                        "Fighter i": fighter_i,
                        "Fighter j": fighter_j,
                        "Fighter i Win %": wins_i / N_CHAINS,
                        "Fighter j Win %": wins_j / N_CHAINS,
                    }
                else:
                    outcome = {
                        "Weight Class": weight_class,
                        "Fighter i": fighter_j,
                        "Fighter j": fighter_i,
                        "Fighter i Win %": wins_j / N_CHAINS,
                        "Fighter j Win %": wins_i / N_CHAINS,
                    }

                weight_results.append(outcome)
                logging.info(f"{fighter_i} vs {fighter_j} -> i: {outcome['Fighter i Win %']:.3f}, j: {outcome['Fighter j Win %']:.3f}")

            except Exception as e:
                logging.error(f"[ERROR] Failed {fighter_i} vs {fighter_j}: {e}")

        # Save results for this weight class
        if weight_results:
            results.extend(weight_results)
            df = pd.DataFrame(weight_results)
            sanitized = weight_class.replace(" ", "_").replace("/", "_")
            file_path = os.path.join(predictions_dir, f"fight_predictions_{sanitized}_{run_id}.xlsx")
            df.to_excel(file_path, index=False)

    except Exception as e:
        logging.error(f"[ERROR] Failed to process {weight_class}: {e}")

# Save combined results
if results:
    pd.DataFrame(results).to_excel(
        os.path.join(predictions_dir, f"fight_predictions_all_{run_id}.xlsx"),
        index=False
    )