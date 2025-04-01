from pathlib import Path
from sklearn.cluster import KMeans

import numpy as np
import pandas as pd

script_dir = Path(__file__).resolve().parent

data = pd.read_excel(script_dir / "data" / "MMAMatches.xlsx")
# Drop rows where Weight Class (lbs) is NaN
data = data.dropna(subset=["Weight Class (lbs)"])
data = data.iloc[:100]