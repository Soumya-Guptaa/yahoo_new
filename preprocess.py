import pandas as pd 
df = pd.read_csv('output_rank.csv')
df['document_no'] = df.groupby('query_id').cumcount() + 1
df.to_csv('output_document.csv', index=False)
text_file_path = 'predictions_train.txt'  
text_df = pd.read_csv(text_file_path, header=None, names=['score'])

csv_file_path = 'output_document.csv'  
csv_df = pd.read_csv(csv_file_path)
if len(csv_df) == len(text_df):
    
    csv_df['score'] = text_df['score']
else:
    print("Error: The number of rows in the text file and CSV file do not match.")
output_file_path = 'output_document.csv'  
csv_df.to_csv(output_file_path, index=False)
df = pd.read_csv(output_file_path)
df_sorted = df.sort_values(by=['query_id', 'score'], ascending=[True, False])
df_sorted.to_csv(output_file_path, index=False)
df = pd.read_csv('output_document.csv')
grouped = df.groupby('query_id').filter(lambda x: len(x) >= 10)

# Save the filtered DataFrame to a new CSV file
output_file_path = 'output10.csv'  # Replace with your desired output file path
grouped.to_csv(output_file_path, index=False)
df = pd.read_csv('output10.csv')

# Select the first 10 rows for each 'query_id'
top_10_per_query_id = df.groupby('query_id').head(10)
top_10_per_query_id.to_csv(output_file_path, index=False)


