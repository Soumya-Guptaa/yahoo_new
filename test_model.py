# import torch
# import pickle
# import numpy as np
# import pandas as pd
# from model import hbiascorrect

# #from main import seq_len, seq_dim, g_embed_dim, f_embed_dim, output_dim, num_layers, num_heads, hidden_size, dropout_rate, train_epoch

# from tqdm import tqdm 

# import matplotlib.pyplot as plt
# from matplotlib import rcParams

# import csv

# def load_data_from_files():
#     with open(f"test_tensor.pkl", 'rb') as f:
#         test_tensor = pickle.load(f)

#     with open(f"test_real_tensor.pkl", 'rb') as f:
#         test_real_tensor = pickle.load(f)

#     return test_tensor, test_real_tensor




# def hit_rate(predictions, targets, top_k=None, epsilon = 1e-3):

#     # Ensure predictions and targets are on the same device
#     device = torch.device("cpu")
#     predictions = predictions.to(device)
#     targets = targets.to(device)


#     # Get the top-k predictions and corresponding targets if top_k is specified
#     if top_k is not None:
#         _, top_indices = torch.topk(predictions, k=top_k, dim=-1)
#         predictions = torch.gather(predictions, -1, top_indices)
#         targets = torch.gather(targets, -1, top_indices)

#     # Sort predictions and targets in descending order
#     _, indices = torch.sort(predictions, descending=True, dim=-1)
#     predictions_sorted = torch.gather(predictions, -1, indices)
#     targets_sorted = torch.gather(targets, -1, indices)
    
#     thresh_relv = 2
#     hit_vector = (targets_sorted >= thresh_relv).to(torch.int)

#     hit_rate = torch.mean(hit_vector.float())
    
# #    print("predictions_sorted")
# #    print(predictions_sorted)
# #
# #    print("targets_sorted")
# #    print(targets_sorted)
# #
# #    print("hit_vector")
# #    print(hit_vector)
# #
# #    input("vector check")

#     return hit_rate.item()



# def average_hit_rate(predictions, targets, top_k=None):
#     """
#     Calculate the average hit rate across all queries.

#     Args:
#     - predictions: Tensor containing predicted relevance scores for all queries.
#     - targets: Tensor containing true relevance scores for all queries.
#     - top_k: Optional parameter to consider only the top-k predictions. Default is None (use all predictions).

#     Returns:
#     - avg_ndcg: Average hit rate score.
#     """

#     # Ensure predictions and targets have the same dimensions
#     assert predictions.size() == targets.size(), "Predictions and targets must have the same dimensions."

#     # Number of queries
#     num_queries = predictions.size(0)

#     # Calculate NDCG for each query
#     query_hit_rate = torch.zeros(num_queries)
#     for i in range(num_queries):
#         query_hit_rate[i] = hit_rate(predictions[i], targets[i], top_k)
    
#     # Calculate average NDCG
#     avg_hit_rate = torch.mean(query_hit_rate)

#     #print("num_queries: ", num_queries)
#     #print("avg_hit_rate: ", avg_hit_rate)

#     return avg_hit_rate.item()






# def ndcg_score(predictions, targets, top_k=None, epsilon = 1e-3):
#     """
#     Calculate Normalized Discounted Cumulative Gain (NDCG) for a given set of predictions and targets.

#     Args:
#     - predictions: Tensor containing predicted relevance scores.
#     - targets: Tensor containing true relevance scores.
#     - top_k: Optional parameter to consider only the top-k predictions. Default is None (use all predictions).

#     Returns:
#     - ndcg: NDCG score.
#     """

#     # Ensure predictions and targets are on the same device
#     device = torch.device("cpu")
#     predictions = predictions.to(device)
#     targets = targets.to(device)


#     # Get the top-k predictions and corresponding targets if top_k is specified
#     if top_k is not None:
#         _, top_indices = torch.topk(predictions, k=top_k, dim=-1)
#         predictions = torch.gather(predictions, -1, top_indices)
#         targets = torch.gather(targets, -1, top_indices)

#     # Sort predictions and targets in descending order
#     _, indices = torch.sort(predictions, descending=True, dim=-1)
#     predictions_sorted = torch.gather(predictions, -1, indices)
#     targets_sorted = torch.gather(targets, -1, indices)

#     # Calculate gains and discounts
#     gains = 2**targets_sorted - 1

#     # add 2 instead of one due to array indices starting at zero
#     discounts = 1 / torch.log2(torch.arange(len(predictions_sorted), dtype=torch.float) + 2)

#     # Handle the exception gains are zero
#     #gains[gains == 0] = epsilon

#     # Calculate DCG
#     dcg = torch.sum(gains * discounts)

#     # Calculate Ideal DCG (IDCG)
#     ideal_gains = 2**torch.sort(targets_sorted, descending=True, dim=-1)[0] - 1
#     idcg = torch.sum(ideal_gains * discounts[:len(ideal_gains)])

#     # Handle the exception ideal_gains are zero
#     if idcg == 0:
#         ndcg = torch.tensor(0)
#     else:
#         # Calculate NDCG
#         ndcg = dcg / idcg


#     return ndcg.item()



# def average_ndcg_score(predictions, targets, top_k=None):
#     """
#     Calculate the average NDCG score across all queries.

#     Args:
#     - predictions: Tensor containing predicted relevance scores for all queries.
#     - targets: Tensor containing true relevance scores for all queries.
#     - top_k: Optional parameter to consider only the top-k predictions. Default is None (use all predictions).

#     Returns:
#     - avg_ndcg: Average NDCG score.
#     """

#     # Ensure predictions and targets have the same dimensions
#     assert predictions.size() == targets.size(), "Predictions and targets must have the same dimensions."

#     # Number of queries
#     num_queries = predictions.size(0)

#     # Calculate NDCG for each query
#     ndcg_scores = torch.zeros(num_queries)
#     for i in range(num_queries):
#         ndcg_scores[i] = ndcg_score(predictions[i], targets[i], top_k)

#     # Calculate average NDCG
#     avg_ndcg = torch.mean(ndcg_scores)

#     #print("num_queries: ", num_queries)
#     #print("avg_ndcg: ", avg_ndcg)

#     return avg_ndcg.item()



# def test_model():

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load the test data
#     print("Loading test data ...")
#     test_tensor, test_real_tensor = load_data_from_files()

#     # Instantiate the model and load the parameters   
    
#     entry_count= test_tensor.size()[0]
#     seq_len= test_tensor.size()[1]
#     seq_dim= test_tensor.size()[-1]
#     output_dim= test_real_tensor.size()[-1]

#     print(f"num_test_entries: {entry_count}, test_seq_len: {seq_len}, test_seq_dim: {seq_dim}, test_output_dim: {output_dim}")

#     num_layers = 3
#     num_heads = 4
#     hidden_size = 256

#     g_embed_dim= 190 
#     f_embed_dim= g_embed_dim - output_dim

#     dropout_rate = 0.1

#     # Load the model parameters
    
#     transformer = hbiascorrect(seq_len, seq_dim, g_embed_dim, f_embed_dim, output_dim, num_layers, num_heads, hidden_size, dropout_rate)
    
#     print("Loading trained model parameters")

#     with open(f"final_model_parameters.pkl", 'rb') as f:
#         model_parameters = pickle.load(f)
#     transformer.load_state_dict(model_parameters)

#     transformer.to(device)
#     transformer.eval()

#     predictions = []
#     targets = []
    
    

#     with torch.no_grad():
#         for tensor, real_tensor in tqdm(zip(test_tensor, test_real_tensor), desc = "test_tensor eval", total = len(test_tensor)):
#             tensor = tensor.unsqueeze(dim=0).to(device)
#             val, _, _ = transformer.forward(tensor)

#             numpy_array = val.cpu().detach().numpy()
#             numpy_array = np.reshape(numpy_array, (-1, seq_len))  
#             predicted_scores = numpy_array
#             predicted_scores = torch.Tensor(predicted_scores)

#             # Append predicted scores and target scores to lists
#             predictions.append(predicted_scores.cpu().numpy().flatten())
#             targets.append(real_tensor.t().cpu().numpy().flatten())

#     # Convert the lists to numpy arrays before creating tensors
#     predictions = torch.tensor(np.array(predictions)).to(device)
#     targets = torch.tensor(np.array(targets)).to(device)

#     abs_err = torch.abs(predictions - targets)

#     print(f"predictions.size(): {predictions.size()}")
#     print(f"targets.size(): {targets.size()}")


#     ndcg3 = average_ndcg_score(predictions, targets, top_k= 3)
#     print(f"NDCG@3 Score: {ndcg3:.4f}")

#     ndcg5 = average_ndcg_score(predictions, targets, top_k= 5)
#     print(f"NDCG@5 Score: {ndcg5:.4f}")

#     # NDCG scores :
#     ndcg = average_ndcg_score(predictions, targets, top_k=None)
#     print(f"NDCG@10 Score: {ndcg:.4f}")


#     hit3 = average_hit_rate(predictions, targets, top_k= 3)
#     print(f"HIT@3 Score: {hit3:.4f}")

#     hit5 = average_hit_rate(predictions, targets, top_k= 5)
#     print(f"HIT@5 Score: {hit5:.4f}")

#     # HIT scores :
#     hit = average_hit_rate(predictions, targets, top_k=None)
#     print(f"HIT@10 Score: {hit:.4f}")


    
#     # Calculate the mean and standard deviation for each item in seq_len
#     mean_errors = torch.mean(abs_err, dim=0).cpu().numpy()
#     std_errors = torch.std(abs_err, dim=0).cpu().numpy()
   
#     avg_mae = np.mean(mean_errors)
#     avg_std = np.mean(std_errors)


#     print(f"\nAverage MAE:  {avg_mae:.4f}")
#     print(f"Average Std:  {avg_std:.4f}\n")

#     return ndcg3, ndcg5, ndcg, hit3, hit5, hit, avg_mae, avg_std


#     # Plot the mean and standard deviation for each item using error bars
# #    items = np.arange(1, seq_len + 1)
# #    x_pos = np.arange(len(items))
# #
# #    fig, ax = plt.subplots(figsize=(10, 6))
# #    ax.errorbar(x_pos, mean_errors, yerr=std_errors, fmt='o', ecolor='r', capsize=5, elinewidth=1, markeredgewidth=1, markerfacecolor='blue', markeredgecolor='blue')
# #    ax.set_xlabel('Items', fontsize=14)
# #    ax.set_ylabel('Mean Absolute Error', fontsize=14)
# #    ax.set_title(f"Mean Absolute Error for Amazon Fashion dataset", fontsize=16)
# #    ax.set_xticks(x_pos)
# #    ax.set_xticklabels(items)
# #    ax.grid(axis='y', linestyle='--', alpha=0.7)
# #    plt.savefig(f"MAE_results.png")
# #    print('MAE_results.png saved') 
# #    plt.show()

#     # After generating the plot and before showing it, save the values to a CSV file
# #    output_file = f"MAE_results.csv"

#     # Open the file outside the 'with' statement to keep it open
# #    file = open(output_file, mode='w')
# #    writer = csv.writer(file)
# #    headers = [f"MAE item{i}" for i in range(1, seq_len + 1)] + [f"Std item{i}" for i in range(1, seq_len + 1)]
# #    writer.writerow(headers)

# #    for i in range(seq_len):
# #        writer.writerow([mean_errors[i], std_errors[i]])

# #    writer.writerow(['Average MAE', 'Average Std', '', '', '', '', '', '', '', '', '', '', '', ''])
# #    writer.writerow([np.mean(mean_errors), np.mean(std_errors)])

#     # Close the file after writing is complete
# #    file.close()
    

# #    print(f"test values written in {output_file}")

# if __name__ == "__main__":
#     test_model()
import torch
import pickle
import numpy as np
import pandas as pd
from model import hbiascorrect

#from main import seq_len, seq_dim, g_embed_dim, f_embed_dim, output_dim, num_layers, num_heads, hidden_size, dropout_rate, train_epoch

from tqdm import tqdm 

import matplotlib.pyplot as plt
from matplotlib import rcParams

import csv

def load_data_from_files():
    with open(f"test_tensor.pkl", 'rb') as f:
        test_tensor = pickle.load(f)

    with open(f"test_real_tensor.pkl", 'rb') as f:
        test_real_tensor = pickle.load(f)

    return test_tensor, test_real_tensor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#------------------------Metrics--------------------------------

#------------------------------------------------------------------------------------------
'''
Code from MULTR implementation
URL: https://github.com/rowedenny/MULTR/blob/main/ultra/utils/metrics.py

'''
class RankingMetricKey(object):
    """Ranking metric key strings."""
    # Mean Receiprocal Rank. For binary relevance.
    MRR = 'mrr'

    # Expected Reciprocal Rank
    ERR = 'err'
    MAX_LABEL = 4

    # Average Relvance Position.
    ARP = 'arp'

    # Normalized Discounted Culmulative Gain.
    NDCG = 'ndcg'

    # Discounted Culmulative Gain.
    DCG = 'dcg'

    # Precision. For binary relevance.
    PRECISION = 'precision'

    # Mean Average Precision. For binary relevance.
    MAP = 'map'

    # Ordered Pair Accuracy.
    ORDERED_PAIR_ACCURACY = 'ordered_pair_accuracy'


def _prepare_and_validate_params(labels, predictions, weights=None, topn=None):
    """Prepares and validates the parameters.

    Args:
      labels: A `Tensor` of the same shape as `predictions`. A value >= 1 means a
        relevant example.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
        former case is per-example and the latter case is per-list.
      topn: A list of cutoff for how many examples to consider for this metric.

    Returns:
      (labels, predictions, weights, topn) ready to be used for metric
      calculation.
    """
    weights = 1.0 if weights is None else weights
    example_weights = torch.ones_like(labels) * weights
    assert predictions.shape == example_weights.shape
    assert predictions.shape == labels.shape
    assert predictions.dim() == 2
    list_size = predictions.shape[1]
    if topn is None:
        topn = [list_size]

    topn = [min(n, list_size) for n in topn]

    # All labels should be >= 0. Invalid entries are reset.
    is_label_valid = labels>= 0.
    labels = labels
    labels = torch.where(
        is_label_valid,
        labels,
        torch.zeros_like(labels))
    # is_label_valid = is_label_valid.to(device=device)
    # if predictions.is_cuda:
    #     predictions = predictions.cpu()
    predictions = torch.where(
        is_label_valid, predictions,
        -1e-6 * torch.ones_like(predictions) + torch.min(
            input=predictions, dim=1, keepdim=True).values)
    return labels, predictions, example_weights, topn


def expected_reciprocal_rank(
        labels, predictions, weights=None, topn=None, name=None):
    """Computes expected reciprocal rank (ERR).

    Args:
      labels: A `Tensor` of the same shape as `predictions`. A value >= 1 means a
        relevant example.
      predictions: A `Tensor` with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
      weights: A `Tensor` of the same shape of predictions or [batch_size, 1]. The
        former case is per-example and the latter case is per-list.
      topn: A list of cutoff for how many examples to consider for this metric.
      name: A string used as the name for this metric.

    Returns:
      A metric for the weighted expected reciprocal rank of the batch.
    """
    labels, predictions, weights, topn = _prepare_and_validate_params(
        labels, predictions, weights, topn)
    _, indices = predictions.sort(descending=True, dim=-1)
    sorted_labels = torch.gather(labels, dim=1, index=indices)
    sorted_weights = torch.gather(weights, dim=1, index=indices)
    list_size = sorted_labels.size()[-1]
    pow = torch.as_tensor(2.0, device=device)
    relevance = (torch.pow(pow, sorted_labels) - 1) / \
      torch.pow(pow, torch.as_tensor(RankingMetricKey.MAX_LABEL,device=device))
    non_rel = torch.cumprod(1.0 - relevance, dim=1) / (1.0 - relevance)
    reciprocal_rank = 1.0 / \
      torch.arange(start=1, end=list_size + 1,device=device,dtype=torch.float32)
    mask = [torch.ge(reciprocal_rank, 1.0 / (n + 1)).type(torch.float32) for n in topn]
    reciprocal_rank_topn = [reciprocal_rank * top_n_mask for top_n_mask in mask]
    # ERR has a shape of [batch_size, 1]
    err = [torch.sum(
        relevance * non_rel * reciprocal_rank * sorted_weights, dim=1, keepdim=True) for reciprocal_rank in reciprocal_rank_topn]
    err = torch.stack(err,dim=0)
    # ERR has size [top_N, batch_size, 1]
    return torch.mean(err,dim=1)


#------------------------------------------------------------------------------------------


def hit_rate(predictions, targets, top_k=None, epsilon = 1e-3):

    # Ensure predictions and targets are on the same device
    device = torch.device("cpu")
    predictions = predictions.to(device)
    targets = targets.to(device)


    # Get the top-k predictions and corresponding targets if top_k is specified
    if top_k is not None:
        _, top_indices = torch.topk(predictions, k=top_k, dim=-1)
        predictions = torch.gather(predictions, -1, top_indices)
        targets = torch.gather(targets, -1, top_indices)

    # Sort predictions and targets in descending order
    _, indices = torch.sort(predictions, descending=True, dim=-1)
    predictions_sorted = torch.gather(predictions, -1, indices)
    targets_sorted = torch.gather(targets, -1, indices)
    
    # Select required accuracy parameter
    threshold_relv = 1
    hit_vector = (targets_sorted >= threshold_relv).to(torch.int)

    hit_rate = torch.mean(hit_vector.float())
   
    return hit_rate.item()



def average_hit_rate(predictions, targets, top_k=None):
    """
    Calculate the average hit rate across all queries.

    Args:
    - predictions: Tensor containing predicted relevance scores for all queries.
    - targets: Tensor containing true relevance scores for all queries.
    - top_k: Optional parameter to consider only the top-k predictions. Default is None (use all predictions).

    Returns:
    - avg_ndcg: Average hit rate score.
    """

    # Ensure predictions and targets have the same dimensions
    assert predictions.size() == targets.size(), "Predictions and targets must have the same dimensions."

    # Number of queries
    num_queries = predictions.size(0)

    # Calculate NDCG for each query
    query_hit_rate = torch.zeros(num_queries)
    for i in range(num_queries):
        query_hit_rate[i] = hit_rate(predictions[i], targets[i], top_k)

    # Calculate average NDCG
    avg_hit_rate = torch.mean(query_hit_rate)

    #print("num_queries: ", num_queries)
    #print("avg_hit_rate: ", avg_hit_rate)

    return avg_hit_rate.item()


def ndcg_score(predictions, targets, top_k=None, epsilon = 1e-3):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) for a given set of predictions and targets.

    Args:
    - predictions: Tensor containing predicted relevance scores.
    - targets: Tensor containing true relevance scores.
    - top_k: Optional parameter to consider only the top-k predictions. Default is None (use all predictions).

    Returns:
    - ndcg: NDCG score.
    """

    # Ensure predictions and targets are on the same device
    device = torch.device("cpu")
    predictions = predictions.to(device)
    targets = targets.to(device)


    # Get the top-k predictions and corresponding targets if top_k is specified
    if top_k is not None:
        _, top_indices = torch.topk(predictions, k=top_k, dim=-1)
        predictions = torch.gather(predictions, -1, top_indices)
        targets = torch.gather(targets, -1, top_indices)

    # Sort predictions and targets in descending order
    _, indices = torch.sort(predictions, descending=True, dim=-1)
    predictions_sorted = torch.gather(predictions, -1, indices)
    targets_sorted = torch.gather(targets, -1, indices)

    # Calculate gains and discounts
    gains = 2**targets_sorted - 1

    # add 2 instead of one due to array indices starting at zero
    discounts = 1 / torch.log2(torch.arange(len(predictions_sorted), dtype=torch.float) + 2)

    # Handle the exception gains are zero
    #gains[gains == 0] = epsilon

    # Calculate DCG
    dcg = torch.sum(gains * discounts)

    # Calculate Ideal DCG (IDCG)
    ideal_gains = 2**torch.sort(targets_sorted, descending=True, dim=-1)[0] - 1
    idcg = torch.sum(ideal_gains * discounts[:len(ideal_gains)])

    # Handle the exception ideal_gains are zero
    if idcg == 0:
        ndcg = torch.tensor(0)
    else:
        # Calculate NDCG
        ndcg = dcg / idcg


    return ndcg.item()



def average_ndcg_score(predictions, targets, top_k=None):
    """
    Calculate the average NDCG score across all queries.

    Args:
    - predictions: Tensor containing predicted relevance scores for all queries.
    - targets: Tensor containing true relevance scores for all queries.
    - top_k: Optional parameter to consider only the top-k predictions. Default is None (use all predictions).

    Returns:
    - avg_ndcg: Average NDCG score.
    """

    # Ensure predictions and targets have the same dimensions
    assert predictions.size() == targets.size(), "Predictions and targets must have the same dimensions."

    # Number of queries
    num_queries = predictions.size(0)

    # Calculate NDCG for each query
    ndcg_scores = torch.zeros(num_queries)
    for i in range(num_queries):
        ndcg_scores[i] = ndcg_score(predictions[i], targets[i], top_k)

    # Calculate average NDCG
    avg_ndcg = torch.mean(ndcg_scores)

    #print("num_queries: ", num_queries)
    #print("avg_ndcg: ", avg_ndcg)

    return avg_ndcg.item()


#----------------------------------------------------------------------------


def test_model():

    # Load the test data
    print("Loading test data ...")
    test_tensor, test_real_tensor = load_data_from_files()

    # Instantiate the model and load the parameters   
    
    entry_count= test_tensor.size()[0]
    seq_len= test_tensor.size()[1]
    seq_dim= test_tensor.size()[-1]
    output_dim= test_real_tensor.size()[-1]

    print(f"num_test_entries: {entry_count}, test_seq_len: {seq_len}, test_seq_dim: {seq_dim}, test_output_dim: {output_dim}")

    num_layers = 3
    num_heads = 4
    hidden_size = 256

    g_embed_dim= 190 
    f_embed_dim= g_embed_dim - output_dim

    dropout_rate = 0.1

    # Load the model parameters
    
    transformer = hbiascorrect(seq_len, seq_dim, g_embed_dim, f_embed_dim, output_dim, num_layers, num_heads, hidden_size, dropout_rate)
    
    print("Loading trained model parameters")

    with open(f"final_model_parameters.pkl", 'rb') as f:
        model_parameters = pickle.load(f)
    transformer.load_state_dict(model_parameters)

    transformer.to(device)
    transformer.eval()

    predictions = []
    targets = []
    
    

    with torch.no_grad():
        for tensor, real_tensor in tqdm(zip(test_tensor, test_real_tensor), desc = "test_tensor eval", total = len(test_tensor)):
            tensor = tensor.unsqueeze(dim=0).to(device)
            val, _, _ = transformer.forward(tensor)

            numpy_array = val.cpu().detach().numpy()
            numpy_array = np.reshape(numpy_array, (-1, seq_len))  
            predicted_scores = numpy_array
            predicted_scores = torch.Tensor(predicted_scores)

            # Append predicted scores and target scores to lists
            predictions.append(predicted_scores.cpu().numpy().flatten())
            targets.append(real_tensor.t().cpu().numpy().flatten())

    # Convert the lists to numpy arrays before creating tensors
    predictions = torch.tensor(np.array(predictions)).to(device)
    targets = torch.tensor(np.array(targets)).to(device)

    abs_err = torch.abs(predictions - targets)

    print(f"predictions.size(): {predictions.size()}")
    print(f"targets.size(): {targets.size()}")


 
    errs= expected_reciprocal_rank( targets, predictions, topn= [1,3,5,10])
    err1, err3, err5, err = [err.item() for err in errs]

    print(f"ERR@1 Score: {err1:.4f}")
    print(f"ERR@3 Score: {err3:.4f}")
    print(f"ERR@5 Score: {err5:.4f}")
    print(f"ERR@10 Score: {err:.4f}")


    ndcg3 = average_ndcg_score(predictions, targets, top_k= 3)
    print(f"NDCG@3 Score: {ndcg3:.4f}")

    ndcg5 = average_ndcg_score(predictions, targets, top_k= 5)
    print(f"NDCG@5 Score: {ndcg5:.4f}")

    # NDCG scores :
    ndcg = average_ndcg_score(predictions, targets, top_k=None)
    print(f"NDCG@10 Score: {ndcg:.4f}")


    hit3 = average_hit_rate(predictions, targets, top_k= 3)
    print(f"HIT@3 Score: {hit3:.4f}")

    hit5 = average_hit_rate(predictions, targets, top_k= 5)
    print(f"HIT@5 Score: {hit5:.4f}")

    # HIT scores :
    hit = average_hit_rate(predictions, targets, top_k=None)
    print(f"HIT@10 Score: {hit:.4f}")


   
    # Calculate the mean and standard deviation for each item in seq_len
    mean_errors = torch.mean(abs_err, dim=0).cpu().numpy()
    std_errors = torch.std(abs_err, dim=0).cpu().numpy()
   
    avg_mae = np.mean(mean_errors)
    avg_std = np.mean(std_errors)


    print(f"\nAverage MAE:  {avg_mae:.4f}")
    print(f"Average Std:  {avg_std:.4f}\n")

    return ndcg3, ndcg5, ndcg, hit3, hit5, hit, err1, err3, err5, err, avg_mae, avg_std


if __name__ == "__main__":
    test_model()





