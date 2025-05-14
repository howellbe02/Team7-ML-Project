from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

matches_data = pd.read_excel("data/MMAMatches.xlsx")

columns = [
    'Fighter i ID', 
    'Fighter j ID',
    'Head Shots Landed', 
    'Head Shots Attempted', 
    'Body Shots Landed', 
    'Body Shots Attempted',
    'Leg Shots Landed',
    'Leg Shots Attempted',
    'Takedowns',
    'Takedown Attempts',
    'Ground Shots Landed',
    'Ground Shots Attempted'
]
fighter_data = matches_data[columns].copy()

fighter_off_averages = fighter_data.groupby('Fighter i ID').mean().reset_index()
fighter_off_averages = fighter_off_averages.drop(columns=['Fighter j ID'])
fighter_off_averages = fighter_off_averages.rename(columns={'Fighter i ID': 'Fighter ID'})

fighter_def_averages = fighter_data.groupby('Fighter j ID').mean().reset_index()
fighter_def_averages = fighter_def_averages.drop(columns=['Fighter i ID'])

fighter_def_averages.columns = [
    'Fighter ID',
    'Head Shots Absorbed',
    'Head Shots Attempted Against',
    'Body Shots Absorbed',
    'Body Shots Attempted Against',
    'Leg Shots Absorbed',
    'Leg Shots Attempted Against',
    'Takedowns Absorbed',
    'Takedowns Attempted Against',
    'Ground Shots Absorbed',
    'Ground Shots Attempted Against'
]

merged_df = pd.merge(fighter_off_averages, fighter_def_averages, on='Fighter ID', how='inner')

merged_df.to_excel("FighterStats.xlsx", index=False)