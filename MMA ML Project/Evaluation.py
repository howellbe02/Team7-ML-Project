import pandas as pd

# Load the merged predictions file
df = pd.read_excel('./predictions/fight_predictions_with_results.xlsx')

# Ensure correct types
df['Fighter i'] = df['Fighter i'].astype(int)
df['winner id'] = pd.to_numeric(df['winner id'], errors='coerce').astype('Int64')

# Filter valid rows and avoid SettingWithCopyWarning
valid = df[df['winner id'].notna()].copy()

# Predict winner based on Fighter i Win %
valid['predicted_winner'] = valid.apply(
    lambda row: row['Fighter i'] if row['Fighter i Win %'] > 0.5 else row['Fighter j'],
    axis=1
)

# Compute accuracy
accuracy = (valid['predicted_winner'] == valid['winner id']).mean()

print(f"Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
