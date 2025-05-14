import os
import numpy as np
import pandas as pd

def get_binomial_probs(df_path, matchups):
    estimates = [
        "Ground_Body_Accuracy",
        "Ground_Head_Accuracy",
        "Standing_Body_Accuracy",
        "Standing_Head_Accuracy",
        "Submission_Accuracy",
        "Takedown_Accuracy",
        "Knockout_Probability",
    ]

    def inv_logit(x):
        return 1 / (1 + np.exp(-x))

    intercept_path = os.path.join(df_path, "All_Intercepts.xlsx")
    intercept_df = pd.read_excel(intercept_path)
    intercepts = dict(zip(intercept_df["Skill Estimate Name"], intercept_df["Intercept"]))

    probs = {}

    for estimate in estimates:
        offense_df = pd.read_excel(os.path.join(df_path, f"Offense_{estimate}.xlsx"))
        defense_df = pd.read_excel(os.path.join(df_path, f"Defense_{estimate}.xlsx"))

        offense_df["Fighter_ID"] = offense_df["Fighter_ID"].astype(str)
        defense_df["Fighter_ID"] = defense_df["Fighter_ID"].astype(str)

        offense_dict = dict(zip(offense_df["Fighter_ID"], offense_df["Estimate"]))
        defense_dict = dict(zip(defense_df["Fighter_ID"], defense_df["Estimate"]))

        intercept = intercepts.get(estimate.replace("_", " "), 0)

        prob_dict = {}
        for i, j in matchups:
            i_str, j_str = str(i), str(j)
            att = offense_dict.get(i_str, 0)
            dff = defense_dict.get(j_str, 0)
            prob_dict[(i_str, j_str)] = inv_logit(att + dff + intercept)

        probs[estimate] = prob_dict

    return probs

def get_poisson_probs(df_path, matchups):
    estimates = [
        "Standing_Shot_Probability",
        "Ground_Shot_Probability",
        "Takedown_Attempt_Probability",
        "Submission_Attempt_Probability",
    ]

    intercept_path = os.path.join(df_path, "All_Intercepts.xlsx")
    intercept_df = pd.read_excel(intercept_path)
    intercepts = dict(zip(intercept_df["Skill Estimate Name"], intercept_df["Intercept"]))

    probs = {}

    for estimate in estimates:
        offense_df = pd.read_excel(os.path.join(df_path, f"Offense_{estimate}.xlsx"))
        defense_df = pd.read_excel(os.path.join(df_path, f"Defense_{estimate}.xlsx"))

        offense_df["Fighter_ID"] = offense_df["Fighter_ID"].astype(str)
        defense_df["Fighter_ID"] = defense_df["Fighter_ID"].astype(str)

        offense_dict = dict(zip(offense_df["Fighter_ID"], offense_df["Estimate"]))
        defense_dict = dict(zip(defense_df["Fighter_ID"], defense_df["Estimate"]))

        intercept = intercepts.get(estimate.replace("_", " "), 0)

        rate_dict = {}
        for i, j in matchups:
            i_str, j_str = str(i), str(j)
            att = offense_dict.get(i_str, 0)
            dff = defense_dict.get(j_str, 0)
            rate_dict[(i_str, j_str)] = np.exp(att + dff + intercept)

        probs[estimate] = rate_dict

    return probs

def get_ground_head_shot_prob(df_path, fighter_ids):
    offense_df = pd.read_excel(os.path.join(df_path, "Ground_Head_Shot_Probability.xlsx"))
    offense_df["Fighter_ID"] = offense_df["Fighter_ID"].astype(str)

    intercept_df = pd.read_excel(os.path.join(df_path, "All_Intercepts.xlsx"))
    intercepts = dict(zip(intercept_df["Skill Estimate Name"], intercept_df["Intercept"]))
    intercept = intercepts.get("Ground Head Shot Probability", 0)

    offense_dict = dict(zip(offense_df["Fighter_ID"], offense_df["Estimate"]))

    def inv_logit(x):
        return 1 / (1 + np.exp(-x))

    prob_dict = {}
    for fid in fighter_ids:
        fid_str = str(fid)
        est = offense_dict.get(fid_str, 0)
        prob_dict[fid_str] = inv_logit(est + intercept)

    return prob_dict

def get_standing_head_shot_prob(df_path, fighter_ids):
    df = pd.read_excel(os.path.join(df_path, "Standing_Head_Shot_Probability.xlsx"))
    df["Fighter_ID"] = df["Fighter_ID"].astype(str)

    intercept_df = pd.read_excel(os.path.join(df_path, "All_Intercepts.xlsx"))
    intercepts = dict(zip(intercept_df["Skill Estimate Name"], intercept_df["Intercept"]))
    intercept = intercepts.get("Standing Head Shot Probability", 0)

    offense_dict = dict(zip(df["Fighter_ID"], df["Estimate"]))

    prob_dict = {}
    for fid in fighter_ids:
        fid_str = str(fid)
        est = offense_dict.get(fid_str, 0)
        prob_dict[fid_str] = 1 / (1 + np.exp(-(est + intercept)))

    return prob_dict

def get_gc_probs(df_path, matchups):
    intercept_path = os.path.join(df_path, "All_Intercepts.xlsx")
    intercept_df = pd.read_excel(intercept_path)
    intercepts = dict(zip(intercept_df["Skill Estimate Name"], intercept_df["Intercept"]))
    intercept = intercepts.get("Ground Control", 0)

    offense_df = pd.read_excel(os.path.join(df_path, "Offense_Ground_Control.xlsx"))
    defense_df = pd.read_excel(os.path.join(df_path, "Defense_Ground_Control.xlsx"))
    offense_df["Fighter_ID"] = offense_df["Fighter_ID"].astype(str)
    defense_df["Fighter_ID"] = defense_df["Fighter_ID"].astype(str)

    offense_dict = dict(zip(offense_df["Fighter_ID"], offense_df["Estimate"]))
    defense_dict = dict(zip(defense_df["Fighter_ID"], defense_df["Estimate"]))

    ground_control_dict = {}
    standup_dict = {}

    for i, j in matchups:
        i_str, j_str = str(i), str(j)
        off = offense_dict.get(i_str, 0)
        dff = defense_dict.get(j_str, 0)
        control = max(off + dff + intercept, 0.01)
        ground_control_dict[(i_str, j_str)] = control
        standup_dict[(i_str, j_str)] = 1 / control

    return {
        "Ground_Control_Per_TD": ground_control_dict,
        "Standup_From_Control": standup_dict
    }