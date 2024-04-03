# import pandas as pd

# def calculate_click_probability(relevance_score, epsilon, ymax, exam_prob):
#     result = epsilon + (1 - epsilon) * ((2**relevance_score - 1) / (2**ymax - 1))
#     return round(result * exam_prob, 3)

# # Load the data
# df = pd.read_csv('output10.csv')

# # Constants
# epsilon = 0.1
# ymax = 4
# examination_prob = [0.68, 0.61, 0.48, 0.34, 0.28, 0.20, 0.11, 0.10, 0.08, 0.06]

# # Create a list to store the final results
# results = []

# # Iterate over unique query IDs
# for query_id in df['query_id'].unique():
#     query_subset = df[df['query_id'] == query_id]
#     query_result = {'query_id': query_id}
    
#     # Calculate click probabilities for each document in the query
#     for i, (_, row) in enumerate(query_subset.iterrows()):
#         relevance_score = int(row['relevance_score'])
#         click_prob = calculate_click_probability(relevance_score, epsilon, ymax, examination_prob[i])
#         query_result[f'doc_{i+1}'] = row['document_no']
#         query_result[f'click_prob_{i+1}'] = click_prob
#         query_result[f'relevance_score_{i+1}'] = relevance_score
    
#     results.append(query_result)

# # Create a DataFrame from the results and save to a CSV file
# results_df = pd.DataFrame(results)
# results_df.to_csv('final_output.csv', index=False)






# ####################
import pandas as pd
import torch.distributions as dist
import pickle

def calculate_click_probability(relevance_score, epsilon, ymax, exam_prob):
    result = epsilon + (1 - epsilon) * ((2**relevance_score - 1) / (2**ymax - 1))
    return round(result * exam_prob, 3)

def calculate_click_probability2(relevance_score,eps , ymax , exam_prob):
            p_relv = eps + (1 - eps) * ((2 * relevance_score - 1) / (2 * ymax - 1))
            # Combine relevance probability and examination probability
            p_click = exam_prob * p_relv
            # Bernoulli distribution for click probability at this stage
            click_dist = dist.Bernoulli(p_click)
            # Sample click/no-click decision for this stage
            click_sample = click_dist.sample()
            # Store the click decision for this stage
            # click_prob[stage] = click_sample
            

# Convert tensor to integer
            integer_value = int(click_sample.item())

            return integer_value


df = pd.read_csv('output10.csv')

# Constants
epsilon = 0.1
ymax = 4
examination_prob = [0.68, 0.61, 0.48, 0.34, 0.28, 0.20, 0.11, 0.10, 0.08, 0.06]

# Create a list to store the final results
results = []

# Iterate over unique query IDs
for query_id in df['query_id'].unique():
    query_subset = df[df['query_id'] == query_id]
    query_result = {'query_id': query_id}
    past_res = 0 
    # Calculate click probabilities for each document in the query
    for i, (_, row) in enumerate(query_subset.iterrows()):
        relevance_score = int(row['relevance_score'])
        if past_res == 0 :
            click_prob = calculate_click_probability2(relevance_score, epsilon, ymax, examination_prob[i])
            if click_prob == 1 :
                 past_res = 1
        else :
            click_prob = 0 
        query_result[f'doc_{i+1}'] = row['document_no']
        query_result[f'click_prob_{i+1}'] = click_prob
        query_result[f'relevance_score_{i+1}'] = relevance_score
    
    results.append(query_result)

# Create a DataFrame from the results and save to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('final_output.csv', index=False)
