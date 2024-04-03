
import pandas as pd
import tqdm as tqdm
data = []
current_query_id = None
doc_number = 0

count= 0
with open('set1.test.txt', 'r') as file:
    for line in file:
        parts = line.split(' ')
        query_id = (int)(parts[1].split(':')[1])
        features = {f"feature{i}": float(0.0) for i in range(700)}
        
        
         
        if query_id != current_query_id:
                current_query_id = query_id
                doc_number = 1
        else:
                doc_number += 1
        for part in parts[2:] :
                
                feature_no = part.split(':')
                
                value = feature_no[-1]
                features[f'feature{feature_no[0]}'] = value
                
                
        features['query_id'] = query_id
        features['document_number'] = doc_number
        data.append(features)
        count+=1
        print(count)
        
            
        


df = pd.DataFrame(data)  
df.to_csv('features_test.csv', index=False)
