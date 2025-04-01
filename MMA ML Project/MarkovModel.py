import numpy as np
import pandas as pd

# Define state names
states = [
    "standing", "stand_head_or_body_strike_attempt_i", "stand_leg_strike_attempt_i",
    "stand_head_or_body_strike_land_i", "stand_leg_strike_land_i",
    "takedown_attempt_i", "ground_control_i", "ground_strike_attempt_i", "ground_strike_land_i",
    "submission_attempt_i", "submission_victory_i", "knockout_victory_i",
    "stand_head_or_body_strike_attempt_j", "stand_leg_strike_attempt_j",
    "stand_head_or_body_strike_land_j", "stand_leg_strike_land_j",
    "takedown_attempt_j", "ground_control_j", "ground_strike_attempt_j", "ground_strike_land_j",
    "submission_attempt_j", "submission_victory_j", "knockout_victory_j"
]

num_states = len(states)

# Initialize transition matrix with zeros
transition_matrix = np.zeros((num_states, num_states))

# Standing transitions
transition_matrix[0, 0] = 0 # Standing → Standing
transition_matrix[0, 1] = 0  # Standing → Head/Body Strike Attempt (i)
transition_matrix[0, 2] = 0  # Standing → Leg Strike Attempt (i)
transition_matrix[0, 5] = 0  # Standing → Takedown Attempt (i)
transition_matrix[0, 12] = 0  # Standing → Head/Body Strike Attempt (j)
transition_matrix[0, 13] = 0 # Standing → Leg Strike Attempt (j)
transition_matrix[0, 16] = 0 # Standing → Takedown Attempt (j)

# Strike attempts leading to landing or miss
transition_matrix[1, 3] = 0  # Strike Attempt (i) → Strike Land (i)
transition_matrix[1, 0] = 0  # Strike Attempt (i) → Back to Standing
transition_matrix[2, 4] = 0  # Leg Strike Attempt (i) → Leg Strike Land (i)
transition_matrix[2, 0] = 0  # Leg Strike Attempt (i) → Back to Standing
transition_matrix[12, 14] = 0  # Strike Attempt (j) → Strike Land (j)
transition_matrix[12, 0] = 0  # Strike Attempt (j) → Back to Standing
transition_matrix[13, 15] = 0  # Leg Strike Attempt (j) → Leg Strike Land (j)
transition_matrix[13, 0] = 0  # Leg Strike Attempt (j) → Back to Standing

# Strike land leading to knockout or reset
transition_matrix[3, 0] = 0  # Strike Land (i) → Back to Standing
transition_matrix[3, 11] = 0 # Strike Land (i) → Knockout Victory (i)
transition_matrix[4, 0] = 0  # Leg Strike Land (i) → Back to Standing
transition_matrix[14, 0] = 0  # Strike Land (j) → Back to Standing
transition_matrix[14, 22] = 0 # Strike Land (j) → Knockout Victory (j)
transition_matrix[15, 0] = 0  # Leg Strike Land (j) → Back to Standing

# Takedown attempts leading to success or failure
transition_matrix[5, 6] = 0  # Takedown Attempt (i) → Ground Control (i)
transition_matrix[5, 0] = 0  # Takedown Attempt (i) → Back to Standing
transition_matrix[16, 17] = 0 # Takedown Attempt (j) → Ground Control (j)
transition_matrix[16, 0] = 0  # Takedown Attempt (j) → Back to Standing

# Ground control transitions
transition_matrix[6, 7] = 0  # Ground Control (i) → Ground Strike Attempt (i)
transition_matrix[6, 9] = 0  # Ground Control (i) → Submission Attempt (i)
transition_matrix[6, 0] = 0  # Ground Control (i) → Stand-Up to Standing
transition_matrix[17, 18] = 0 # Ground Control (j) → Ground Strike Attempt (j)
transition_matrix[17, 20] = 0 # Ground Control (j) → Submission Attempt (j)
transition_matrix[17, 0] = 0  # Ground Control (j) → Stand-Up to Standing

# Ground strike attempts leading to landing or miss
transition_matrix[7, 8] = 0  # Ground Strike Attempt (i) → Ground Strike Land (i)
transition_matrix[7, 6] = 0  # Ground Strike Attempt (i) → Back to Ground Control (i)
transition_matrix[18, 19] = 0 # Ground Strike Attempt (j) → Ground Strike Land (j)
transition_matrix[18, 17] = 0 # Ground Strike Attempt (j) → Back to Ground Control (j)

# Ground strike land leading to knockout or reset
transition_matrix[8, 6] = 0  # Ground Strike Land (i) → Back to Ground Control (i)
transition_matrix[8, 11] = 0  # Ground Strike Land (i) → Knockout Victory (i)
transition_matrix[19, 17] = 0 # Ground Strike Land (j) → Back to Ground Control (j)
transition_matrix[19, 22] = 0 # Ground Strike Land (j) → Knockout Victory (j)

# Submission attempts leading to victory or failure
transition_matrix[9, 10] = 0  # Submission Attempt (i) → Submission Victory (i)
transition_matrix[9, 6] = 0  # Submission Attempt (i) → Back to Ground Control (i)
transition_matrix[20, 21] = 0 # Submission Attempt (j) → Submission Victory (j)
transition_matrix[20, 17] = 0 # Submission Attempt (j) → Back to Ground Control (j)

# Normalize rows to sum to 1 (for valid Markov property)
row_sums = transition_matrix.sum(axis=1, keepdims=True)
transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0)

