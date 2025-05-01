import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error

# original dataset
df_full = pd.read_excel("data/NewMMAMatches.xlsx")

# Drop null values for these columns for df variable
df = df_full[df_full["Match Length Sec"].notnull() & df_full["Significant Strikes Landed"].notnull()]

df = df.sort_values("Date")

# new data frame for regression with relevant data
fighter_stats = pd.DataFrame()

for role in ["i", "j"]:
    temp = df.copy()
    temp["Fighter ID"] = temp[f"Fighter {role} ID"]
    temp["Sig Strikes"] = temp["Significant Strikes Landed"]
    temp["Minutes"] = temp["Match Length Sec"] / 60
    fighter_stats = pd.concat([fighter_stats, temp[["Fighter ID", "Date", "Sig Strikes", "Minutes"]]])

fighter_stats = fighter_stats.sort_values(["Fighter ID", "Date"])

# Calculate Cumulative Sig Strikes / Cumulative Minutes
fighter_stats["Cumulative Sig Strikes"] = fighter_stats.groupby("Fighter ID")["Sig Strikes"].cumsum().shift()
fighter_stats["Cumulative Minutes"] = fighter_stats.groupby("Fighter ID")["Minutes"].cumsum().shift()
fighter_stats["CSLpCM"] = fighter_stats["Cumulative Sig Strikes"] / fighter_stats["Cumulative Minutes"]

# drop rows where CSLpCM can't be calculated because a fighter had no prior fights (new to UFC)
fighter_stats = fighter_stats.dropna(subset=["CSLpCM"])

# extract relevant data for regression
fight_data = df[["Fighter i ID", "Fighter j ID", "Date", "Win or Loss"]].copy()

# Merge CSLpCM for Fighter i
fight_data = fight_data.merge(
    fighter_stats[["Fighter ID", "Date", "CSLpCM"]],
    left_on=["Fighter i ID", "Date"],
    right_on=["Fighter ID", "Date"],
    how="left"
).rename(columns={"CSLpCM": "CSLpCM_i"}).drop(columns=["Fighter ID"])

# Merge CSLpCM for Fighter j
fight_data = fight_data.merge(
    fighter_stats[["Fighter ID", "Date", "CSLpCM"]],
    left_on=["Fighter j ID", "Date"],
    right_on=["Fighter ID", "Date"],
    how="left"
).rename(columns={"CSLpCM": "CSLpCM_j"}).drop(columns=["Fighter ID"])

# drop fights where a fighter is new to UFC
train_data = fight_data.dropna(subset=["CSLpCM_i", "CSLpCM_j"]).copy()

# Feature: difference in historical striking output
train_data["CSLpCM_Diff"] = train_data["CSLpCM_i"] - train_data["CSLpCM_j"]

# Target: 1 if Fighter i won, 0 otherwise
train_data["Fighter_i_Wins"] = (train_data["Win or Loss"] == "W").astype(int)

# Define inputs for model
X = train_data[["CSLpCM_Diff"]]
y = train_data["Fighter_i_Wins"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict regression
y_pred = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1] 

# Evaluate reg
print(classification_report(y_test, y_pred))
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
mse = mean_squared_error(y_test, probs)
print("Mean Squared Error (MSE): {:.4f}".format(mse))