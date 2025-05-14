import numpy as np
import pandas as pd
import warnings

states = [
    # Neutral state
    "standing",                      # 0

    # Fighter i
    "stand_strike_attempt_i",        # 1
    "stand_body_attempt_i",          # 2
    "stand_head_attempt_i",          # 3
    "stand_body_landed_i",           # 4
    "stand_head_landed_i",           # 5
    "knockout_victory_i",            # 6
    "takedown_attempt_i",            # 7
    "ground_control_i",              # 8
    "submission_attempt_i",          # 9
    "submission_victory_i",          # 10
    "ground_strike_attempt_i",       # 11
    "ground_body_attempt_i",         # 12
    "ground_head_attempt_i",         # 13
    "ground_body_land_i",            # 14
    "ground_head_land_i",            # 15

    # Fighter j
    "stand_strike_attempt_j",        # 16
    "stand_body_attempt_j",          # 17
    "stand_head_attempt_j",          # 18
    "stand_body_landed_j",           # 19
    "stand_head_landed_j",           # 20
    "knockout_victory_j",            # 21
    "takedown_attempt_j",            # 22
    "ground_control_j",              # 23
    "submission_attempt_j",          # 24
    "submission_victory_j",          # 25
    "ground_strike_attempt_j",       # 26
    "ground_body_attempt_j",         # 27
    "ground_head_attempt_j",         # 28
    "ground_body_land_j",            # 29
    "ground_head_land_j"             # 30
]

state_to_index = {state: i for i, state in enumerate(states)}

TERMINAL_STATES = {
    "knockout_victory_i",
    "submission_victory_i",
    "knockout_victory_j",
    "submission_victory_j"
}

num_states = len(states)

# Helper functions to get transition probabilities
def get_binom_prob(probs_dict, category, fighter_i, fighter_j, default=0.5):
    key = (str(int(float(fighter_i))), str(int(float(fighter_j))))
    value = probs_dict.get(category, {}).get(key, None)
    
    if value is None:
        warnings.warn(
            f"Probability missing for category '{category}' with pair ({fighter_i}, {fighter_j}). "
            f"Returning default value {default}."
        )
        return default

    return value

def get_ghp_prob(ghp_dict, fighter_id, default=0.5):
    return ghp_dict.get(str(fighter_id), default)

def get_shp_prob(shp_dict, fighter_id, default=0.5):
    return shp_dict.get(str(fighter_id), default)

def get_poisson_rate(rate_dict, category, fighter_i, fighter_j, default=0.1):
    key = (str(int(float(fighter_i))), str(int(float(fighter_j))))
    value = rate_dict.get(category, {}).get(key, None)

    if value is None:
        warnings.warn(
            f"Poisson rate missing for category '{category}' with pair ({fighter_i}, {fighter_j}). "
            f"Returning default lambda {default}."
        )
        return default

    return value

def get_gamma_prob(prob_dict, fighter_i, fighter_j, default=0.1):
    key = (str(int(float(fighter_i))), str(int(float(fighter_j))))
    value = prob_dict.get(key, None)
    if value is None:
        warnings.warn(
            f"Gamma probability missing for pair ({fighter_i}, {fighter_j}). Returning default value {default}."
        )
        return default
    return value
##############################################


# Generate transition matrix between fighter i and j
def markov_transition_matrix(fighter_i, fighter_j, binomial_probs, poisson_probs, gamma_probs, ghp_probs, shp_probs):
    # Initialize transition matrix with zeros
    transition_matrix = np.zeros((num_states, num_states))

    # Get Poisson rates for all actions from standing
    λ_i_strike   = get_poisson_rate(poisson_probs, "Standing_Shot_Probability", fighter_i, fighter_j)
    λ_i_td       = get_poisson_rate(poisson_probs, "Takedown_Attempt_Probability", fighter_i, fighter_j)
    λ_j_strike   = get_poisson_rate(poisson_probs, "Standing_Shot_Probability", fighter_j, fighter_i)
    λ_j_td       = get_poisson_rate(poisson_probs, "Takedown_Attempt_Probability", fighter_j, fighter_i)

    # Total rate of leaving standing state
    total_rate = λ_i_strike + λ_i_td + λ_j_strike + λ_j_td

    # Assign transition probabilities from standing
    transition_matrix[0, 1]  = λ_i_strike / total_rate  # standing → stand_strike_attempt_i
    transition_matrix[0, 7]  = λ_i_td     / total_rate  # standing → takedown_attempt_i
    transition_matrix[0, 16] = λ_j_strike / total_rate  # standing → stand_strike_attempt_j
    transition_matrix[0, 22] = λ_j_td     / total_rate  # standing → takedown_attempt_j
    transition_matrix[0, 0]  = 0                        # standing → standing (optional: could be a small rest rate)
    
    # i stand strike flow
    transition_matrix[1, 2] = 1 - transition_matrix[1, 3]                                               # strike_attempt_i → body_attempt_i    # ← binomial: Standing_Head_Shot_Probability
    transition_matrix[1, 3] = get_shp_prob(shp_probs, fighter_i)  # strike_attempt_i → head_attempt_i # ← binomial: Standing_Head_Shot_Probability
    transition_matrix[2, 4] = get_binom_prob(binomial_probs, 'Standing_Body_Accuracy', fighter_i, fighter_j)  # body_attempt_i → body_landed_i       # ← binomial: Standing_Body_Accuracy
    transition_matrix[2, 0] = 1 - transition_matrix[2, 4]                                               # body_attempt_i → standing (miss)     # ← binomial: Standing_Body_Accuracy
    transition_matrix[3, 5] = get_binom_prob(binomial_probs, 'Standing_Head_Accuracy', fighter_i, fighter_j)  # head_attempt_i → head_landed_i       # ← binomial: Standing_Head_Accuracy
    transition_matrix[3, 0] = 1 - transition_matrix[3, 5]                                               # head_attempt_i → standing (miss)     # ← binomial: Standing_Head_Accuracy
    transition_matrix[4, 0] = 1                                                                         # body_landed_i → standing
    transition_matrix[5, 0] = 1 - transition_matrix[5, 6]                                               # head_landed_i → standing             # ← binomial: Knockout_Probability
    transition_matrix[5, 6] = get_binom_prob(binomial_probs, "Knockout_Probability", fighter_i, fighter_j)    # head_landed_i → knockout_victory_i   # ← binomial: Knockout_Probability

    # j stand strike flow
    transition_matrix[16, 17] = 1 - transition_matrix[16, 18]                                             # strike_attempt_j → body_attempt_j  # ← binomial: Standing_Head_Shot_Probability
    transition_matrix[16, 18] = get_shp_prob(shp_probs, fighter_j) # strike_attempt_j → head_attempt_j # ← binomial: Standing_Head_Shot_Probability
    transition_matrix[17, 19] = get_binom_prob(binomial_probs, 'Standing_Body_Accuracy', fighter_j, fighter_i)  # body_attempt_j → body_landed_j     # ← binomial: Standing_Body_Accuracy
    transition_matrix[17, 0] =  1 - transition_matrix[17, 19]                                             # body_attempt_j → standing (miss)   # ← binomial: Standing_Body_Accuracy
    transition_matrix[18, 20] = get_binom_prob(binomial_probs, 'Standing_Head_Accuracy', fighter_j, fighter_i)  # head_attempt_j → head_landed_j     # ← binomial: Standing_Head_Accuracy
    transition_matrix[18, 0] = 1 - transition_matrix[18, 20]                                              # head_attempt_j → standing (miss)   # ← binomial: Standing_Head_Accuracy
    transition_matrix[19, 0] = 1                                                                        # body_landed_j → standing
    transition_matrix[20, 0] = 1 - transition_matrix[20, 21]                                            # head_landed_j → standing                # ← binomial: Knockout_Probability
    transition_matrix[20, 21] = get_binom_prob(binomial_probs, "Knockout_Probability", fighter_j, fighter_i)  # head_landed_j → knockout_victory_j      # ← binomial: Knockout_Probability

    # Takedowns
    transition_matrix[7, 8] = get_binom_prob(binomial_probs, 'Takedown_Accuracy', fighter_i, fighter_j)       # takedown_attempt_i → ground_control_i   # ← binomial: Takedown_Accuracy
    transition_matrix[7, 0] = 1 - transition_matrix[7, 8]                                               # takedown_attempt_i → standing (failed)  # ← binomial: Takedown_Accuracy
    transition_matrix[22, 23] = get_binom_prob(binomial_probs, 'Takedown_Accuracy', fighter_j, fighter_i)     # takedown_attempt_j → ground_control_j   # ← binomial: Takedown_Accuracy
    transition_matrix[22, 0] = 1 - transition_matrix[22, 23]                                            # takedown_attempt_j → standing (failed)  # ← binomial: Takedown_Accuracy

    # i ground control options
    λ_sub = get_poisson_rate(poisson_probs, "Submission_Attempt_Probability", fighter_i, fighter_j)
    λ_strike = get_poisson_rate(poisson_probs, "Ground_Shot_Probability", fighter_i, fighter_j)
    λ_i_standup = get_gamma_prob(gamma_probs["Standup_From_Control"], fighter_i, fighter_j)
    λ_self = 0
    total_i = λ_sub + λ_strike + λ_i_standup + λ_self

    transition_matrix[8, 9] = λ_sub / total_i   # ground_control_i → submission_attempt_i
    transition_matrix[8, 11] = λ_strike / total_i  # ground_control_i → ground_strike_attempt_i
    transition_matrix[8, 0] = λ_i_standup / total_i   # stand up from ground_control_i
    transition_matrix[8, 8] = λ_self / total_i   # ground_control_i → ground_control_i

    # i submission
    transition_matrix[9, 10] = get_binom_prob(binomial_probs, 'Submission_Accuracy', fighter_i, fighter_j)  # submission_attempt_i → submission_victory_i   # ← binomial: Submission_Accuracy
    transition_matrix[9, 8] = 1 - transition_matrix[9, 10]                                            # submission_attempt_i → back to control (fail) # ← binomial: Submission_Accuracy

    # i ground striking
    transition_matrix[11, 12] = 1 - transition_matrix[11, 13]                                           # ground_strike_attempt_i → body_attempt_i   # ← binomial: Ground_Shot_Probability
    transition_matrix[11, 13] = get_poisson_rate(poisson_probs, "Ground_Shot_Probability", fighter_i, fighter_j) # ground_strike_attempt_i → head_attempt_i # ← binomial: Ground_Shot_Probability
    transition_matrix[12, 14] = get_binom_prob(binomial_probs, 'Ground_Body_Accuracy', fighter_i, fighter_j)  # body_attempt_i → body_land_i               # ← binomial: Ground_Body_Accuracy
    transition_matrix[12, 8] = 1 - transition_matrix[12, 14]                                            # body_attempt_i → miss → back to control    # ← binomial: Ground_Body_Accuracy
    transition_matrix[13, 15] = get_ghp_prob(ghp_probs, fighter_i)                                      # head_attempt_i → head_land_i               # ← binomial: Ground_Head_Accuracy
    transition_matrix[13, 8] = 1 - transition_matrix[13, 15]                                            # head_attempt_i → miss → back to control    # ← binomial: Ground_Head_Accuracy
    transition_matrix[14, 8] = 1  # body_land_i → back to control
    transition_matrix[15, 8] = 1 - transition_matrix[15, 6]                                             # head_land_i → back to control              # ← binomial: Knockout_Probability
    transition_matrix[15, 6] = get_binom_prob(binomial_probs, "Knockout_Probability", fighter_i, fighter_j)   # head_land_i → knockout_victory_i           # ← binomial: Knockout_Probability

    # j ground control options
    λ_j_sub      = get_poisson_rate(poisson_probs, "Submission_Attempt_Probability", fighter_j, fighter_i)
    λ_j_gstrike  = get_poisson_rate(poisson_probs, "Ground_Shot_Probability", fighter_j, fighter_i)
    λ_j_standup = get_gamma_prob(gamma_probs["Standup_From_Control"], fighter_j, fighter_i)
    λ_j_self = 0
    total_j = λ_j_sub + λ_j_gstrike + λ_j_standup + λ_j_self

    transition_matrix[23, 24] = λ_j_sub / total_j # ground_control_j → submission_attempt_j
    transition_matrix[23, 26] = λ_j_gstrike / total_j # ground_control_j → ground_strike_attempt_j
    transition_matrix[23, 0] = λ_j_standup / total_j  # stand up from ground_control_j
    transition_matrix[23, 23] = λ_j_self / total_j # ground_control_j → ground_control_j

    # j submission
    transition_matrix[24, 25] = get_binom_prob(binomial_probs, 'Submission_Accuracy', fighter_j, fighter_i)   # submission_attempt_j → submission_victory_j   # ← binomial: Submission_Accuracy
    transition_matrix[24, 23] = 1 - transition_matrix[24, 25]                                           # submission_attempt_j → fail → back to control # ← binomial: Submission_Accuracy

    # j ground striking
    transition_matrix[26, 27] = 1 - transition_matrix[26, 28]                                           # ground_strike_attempt_j → body_attempt_j      # ← binomial: Ground_Shot_Probability
    transition_matrix[26, 28] = get_poisson_rate(poisson_probs, "Ground_Shot_Probability", fighter_j, fighter_i) # ground_strike_attempt_j → head_attempt_j    # ← binomial: Ground_Shot_Probability
    transition_matrix[27, 29] = get_binom_prob(binomial_probs, 'Ground_Body_Accuracy', fighter_j, fighter_i)  # body_attempt_j → body_land_j                  # ← binomial: Ground_Body_Accuracy
    transition_matrix[27, 23] = 1 - transition_matrix[27, 29]                                           # body_attempt_j → miss → back to control       # ← binomial: Ground_Body_Accuracy
    transition_matrix[28, 30] = get_ghp_prob(ghp_probs, fighter_j)                                      # head_attempt_j → head_land_j                  # ← binomial: Ground_Head_Accuracy
    transition_matrix[28, 23] = 1 - transition_matrix[28, 30]                                           # head_attempt_j → miss → back to control       # ← binomial: Ground_Head_Accuracy
    transition_matrix[29, 23] = 1 # body_land_j → back to control
    transition_matrix[30, 23] = 1 - transition_matrix[30, 21]                                           # head_land_j → back to control                 # ← binomial: Knockout_Probability
    transition_matrix[30, 21] = get_binom_prob(binomial_probs, "Knockout_Probability", fighter_j, fighter_i)  # head_land_j → knockout_victory_j              # ← binomial: Knockout_Probability


    # Normalize rows to sum to 1
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0)

    return transition_matrix

def simulate_chain(transition_matrix, states, initial_state="standing", max_steps=900, state_to_index=None):
    if state_to_index is None:
        state_to_index = {state: i for i, state in enumerate(states)}

    current_index = state_to_index[initial_state]
    chain = [initial_state]

    metrics = {
        "strikes_landed_i": 0,
        "strikes_landed_j": 0,
        "takedowns_i": 0,
        "takedowns_j": 0,
        "submissions_i": 0,
        "submissions_j": 0,
        "control_time_i": 0,
        "control_time_j": 0
    }

    for _ in range(max_steps):
        probs = transition_matrix[current_index]
        next_index = np.random.choice(len(states), p=probs)
        next_state = states[next_index]
        chain.append(next_state)


        # Metrics tracking
        if next_state in {"stand_body_landed_i", "stand_head_landed_i", "ground_body_land_i", "ground_head_land_i"}:
            metrics["strikes_landed_i"] += 1
        if next_state in {"stand_body_landed_j", "stand_head_landed_j", "ground_body_land_j", "ground_head_land_j"}:
            metrics["strikes_landed_j"] += 1
        if next_state == "ground_control_i":
            metrics["takedowns_i"] += 1 if chain[-2] == "takedown_attempt_i" else 0
            metrics["control_time_i"] += 1
        if next_state == "ground_control_j":
            metrics["takedowns_j"] += 1 if chain[-2] == "takedown_attempt_j" else 0
            metrics["control_time_j"] += 1
        if next_state == "submission_attempt_i":
            metrics["submissions_i"] += 1
        if next_state == "submission_attempt_j":
            metrics["submissions_j"] += 1

        if next_state in TERMINAL_STATES:
            break

        current_index = next_index

    return chain, metrics

def decision_probability(
    delta_al,    # difference in strikes landed
    delta_tdl,   # difference in takedowns landed
    delta_sma,   # difference in submission attempts
    delta_ctrl   # difference in control time
):
    # Logistic regression coefficients
    beta_1 = 0.0716  # for ΔAL
    beta_2 = 0.1334  # for ΔTDL
    beta_3 = 0.1861  # for ΔSMA
    beta_4 = 0.0035  # for ΔCTRL

    logit = (
        beta_1 * delta_al +
        beta_2 * delta_tdl +
        beta_3 * delta_sma +
        beta_4 * delta_ctrl
    )
    prob = 1 / (1 + np.exp(-logit))  # inverse logit
    return prob


def determine_winner(event_sequence, metrics):
    # Terminal states for finishes
    if "knockout_victory_i" in event_sequence or "submission_victory_i" in event_sequence:
        return "fighter_i"
    elif "knockout_victory_j" in event_sequence or "submission_victory_j" in event_sequence:
        return "fighter_j"
    
    # Judging simulation
    delta_al = metrics["strikes_landed_i"] - metrics["strikes_landed_j"]
    delta_tdl = metrics["takedowns_i"] - metrics["takedowns_j"]
    delta_sma = metrics["submissions_i"] - metrics["submissions_j"]
    delta_ctrl = metrics["control_time_i"] - metrics["control_time_j"]

    p_win_i = decision_probability(delta_al, delta_tdl, delta_sma, delta_ctrl)

    return "fighter_i" if np.random.rand() < p_win_i else "fighter_j"