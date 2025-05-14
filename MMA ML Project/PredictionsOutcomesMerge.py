import pandas as pd

# Paths
PRED_PATH   = "./predictions/fight_predictions_all_1.xlsx"
RAW_PATH    = "./data/NewMMAMatches.xlsx"
OUTPUT_PATH = "./predictions/fight_predictions_with_results.xlsx"

# 1) Load both files
pred_df = pd.read_excel(PRED_PATH)
raw_df  = pd.read_excel(RAW_PATH)

# 2) Normalize
pred_df["Fighter i"]    = pred_df["Fighter i"].astype(int)
pred_df["Fighter j"]    = pred_df["Fighter j"].astype(int)
pred_df["Weight Class"] = pred_df["Weight Class"].astype(str).str.strip()

raw_df["Fighter i ID"]         = raw_df["Fighter i ID"].astype(int)
raw_df["Fighter j ID"]         = raw_df["Fighter j ID"].astype(int)
raw_df["Cleaned Weight Class"] = raw_df["Cleaned Weight Class"].astype(str).str.strip()
raw_df["raw_outcome"] = (
    raw_df["Win or Loss"]
      .astype(str)
      .str.strip()
      .str.upper()
      .replace({"W":"W","L":"L","D":"D","NC":"NC"})
)

# 3) Compute winner id
def get_winner_id(row):
    if row["raw_outcome"] == "W":
        return row["Fighter i ID"]
    elif row["raw_outcome"] == "L":
        return row["Fighter j ID"]
    else:
        return pd.NA

raw_df["winner id"] = raw_df.apply(get_winner_id, axis=1)

# 4) Build an ordered lookup
lookup = {
    (r["Fighter i ID"], r["Fighter j ID"], r["Cleaned Weight Class"]): 
      (r["winner id"], r["raw_outcome"], r["Date"])
    for _, r in raw_df.iterrows()
}

# 5) Fill predictions
winner_ids = []
results    = []
dates      = []

for _, p in pred_df.iterrows():
    i, j = int(p["Fighter i"]), int(p["Fighter j"])
    wc   = p["Weight Class"]
    entry = lookup.get((i, j, wc)) or lookup.get((j, i, wc))
    
    if entry:
        win_id, raw_out, fight_date = entry
        if raw_out in ("D", "NC"):
            res = raw_out
        elif win_id == i:
            res = "W"
        else:
            res = "L"
    else:
        win_id     = pd.NA
        res        = pd.NA
        fight_date = pd.NaT

    winner_ids.append(win_id)
    results.append(res)
    dates.append(fight_date)

pred_df["winner id"] = winner_ids
pred_df["Result"]    = results
pred_df["Date"]      = dates

# 6) Save the file
pred_df.to_excel(OUTPUT_PATH, index=False)
print(f"Saved results to {OUTPUT_PATH}")
