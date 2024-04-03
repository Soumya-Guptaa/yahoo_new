import pandas as pd 
input_file_path = 'train_filtered.txt'
output = 'output_rank.csv'

df = pd.DataFrame(columns=['query_id', 'relevance_score'])
count = 0
with open(input_file_path, 'r') as file:
    for line in file:
        # Split the line into entries
        relevance = line.split(' ')[0]
        qid = line.split(' ')[1].split(':')[1]


        
        new_row = pd.DataFrame({'query_id': [qid], 'relevance_score': [relevance]})
        df = pd.concat([df, new_row], ignore_index=True)
        print(count)
        count = count+1

df.to_csv(output, index=False)
        
        


        

