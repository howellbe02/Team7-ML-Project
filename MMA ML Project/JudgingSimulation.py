import pandas as pd
import numpy as np
import statsmodels.api as sm

file_path = "C:/Users/danie/Desktop/Test MMA Project/MMA ML Project/data/NewMMAMatches.xlsx"
df = pd.read_excel(file_path)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Win or Loss"] = df["Win or Loss"].astype(str).str.strip().str.upper()
df["Win Type"] = df["Win Type"].astype(str).str.strip()

df = df[
    (df["Date"] <= pd.Timestamp("2023-05-05")) &
    (df["Win Type"].str.contains("Decision", case=False, na=False))
].reset_index(drop=True)

rows = []
for i in range(0, len(df), 2):
    f1 = df.iloc[i]
    f2 = df.iloc[i + 1]

    if f1["Win or Loss"] == "W":
        winner = f1
        loser = f2
    else:
        winner = f2
        loser = f1

    rows.append({
        "Delta_Strikes": winner["Significant Strikes Landed"] - loser["Significant Strikes Landed"],
        "Delta_Takedowns": winner["Takedowns"] - loser["Takedowns"],
        "Delta_Submissions": winner["Submission Attempts"] - loser["Submission Attempts"],
        "Delta_Control": winner["Control Time Sec"] - loser["Control Time Sec"]
    })

X = pd.DataFrame(rows)
y = pd.Series([1] * len(X))

X2 = -X.copy()
y2 = pd.Series([0] * len(X2))

X_full = pd.concat([X, X2], ignore_index=True)
y_full = pd.concat([y, y2], ignore_index=True)

X_full = X_full.replace([np.inf, -np.inf], np.nan)
mask = X_full.notna().all(axis=1) & y_full.notna()
X_full = X_full[mask]
y_full = y_full[mask]

model = sm.Logit(y_full, X_full).fit()
print(model.summary())
