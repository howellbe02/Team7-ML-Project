import numpy as np
import pandas as pd
from scipy.optimize import minimize

data = pd.read_excel("C:/Users/danie/Desktop/Betting Simulation/MMAMatches.xlsx")
# Drop rows where Weight Class (lbs) is NaN
data = data.dropna(subset=["Weight Class (lbs)"])
data = data.iloc[:100]

# Fighter list
fighters = list(set(data["Fighter i ID"]).union(set(data["Fighter j ID"])))
n_fighters = len(fighters)

# Initialize attack, defense, and weight parameters
np.random.seed(42)
attack = np.random.normal(0, 1, n_fighters)
defense = np.random.normal(0, 1, n_fighters)
weight_effect = 0
o = 0
# Fighter ID mapping
fighter_idx = {fighter: i for i, fighter in enumerate(fighters)}

# Log-Likelihood function for Poisson Model with Duration Offset
def poisson_log_likelihood(params, data):
    global o
    attack = params[:n_fighters]
    defense = params[n_fighters:2*n_fighters]
    weight_effect = params[-1]
    log_likelihood = 0
    for _, row in data.iterrows():
        i = fighter_idx[row["Fighter i ID"]]
        j = fighter_idx[row["Fighter j ID"]]
        weight = row["Weight Class (lbs)"]
        duration = row["Match Length Sec"]
        
        # Poisson rate (log-linear model with log(duration) offset)
        lambda_ij = np.exp(attack[i] - defense[j] + weight_effect * weight) * (1/duration)
        
        # Poisson log-likelihood
        log_likelihood += row["Standing Head and Body Shots Attempted"] * np.log(lambda_ij) - lambda_ij
        print(o)
        o += 1
    return -log_likelihood  # Negative because we minimize

# EM Algorithm
def expectation_maximization(data, max_iter=100, tol=1e-4):
    global attack, defense, weight_effect
    params = np.concatenate([attack, defense, [weight_effect]])
    for iteration in range(max_iter):
        prev_params = params.copy()
        
        # Maximization Step: Find new parameters that maximize likelihood
        result = minimize(poisson_log_likelihood, params, args=(data,), method="L-BFGS-B", options={"maxiter": 10})
        params = result.x  # Update parameters
        
        # Update attack, defense, and weight effect
        attack, defense, weight_effect = params[:n_fighters], params[n_fighters:2*n_fighters], params[-1]

        # Save results after each iteration
        results_df = pd.DataFrame({
            "Fighter ID": fighters,
            "Attack Ability": attack,
            "Defense Ability": defense
        })

        weight_effect_df = pd.DataFrame({"Weight Effect": [weight_effect]})

        with pd.ExcelWriter("MMA_Fighter_Skills.xlsx") as writer:
            results_df.to_excel(writer, sheet_name="Fighter Skills", index=False)
            weight_effect_df.to_excel(writer, sheet_name="Weight Effect", index=False)

        print(f"Iteration {iteration+1} saved to Excel.")

        # Check convergence
        if np.linalg.norm(params - prev_params) < tol:
            print(f"Converged in {iteration+1} iterations.")
            break
    
    return params[:n_fighters], params[n_fighters:2*n_fighters], params[-1]

# Run EM
attack, defense, weight_effect = expectation_maximization(data)

# Print results
for i, fighter in enumerate(fighters):
    print(f"Fighter {fighter}: Attack = {attack[i]:.2f}, Defense = {defense[i]:.2f}")

print(f"Weight Class Effect: {weight_effect:.4f}")
