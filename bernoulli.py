import pandas as pd
import numpy as np
df = pd.read_csv('final_output.csv')
relevance_score_columns = [col for col in df.columns if "relevance_score" in col]
print(relevance_score_columns)
for col in relevance_score_columns:
    df[col] = (df[col]) / (4)


df.to_csv('final_1_output.csv', index=False)