import pandas as pd
import numpy as np
import torch
from train_feature import get_tensor
from train_feature2 import get_tensor2

from sklearn.model_selection import train_test_split

import pickle

seq_len = 10
feature_dim = 701
X , Y , Y_real = get_tensor()
X_test , Y_test , Y_real_test = get_tensor2()
print(X.shape)
X_train = torch.tensor(X.values).view(-1, seq_len, feature_dim)
X_test = torch.tensor(X_test.values).view(-1, seq_len, feature_dim)
print(Y.shape)
print(Y_real.shape)
# Check the shape of the tensor

# Reshape the ml dataset into a tensor
y_train = torch.tensor(Y.values, dtype=torch.float32).view(-1, seq_len)
y_real_train = torch.tensor(Y_real.values, dtype=torch.float32).view(-1, seq_len)
y_test =  torch.tensor(Y_test.values, dtype=torch.float32).view(-1, seq_len)
y_real_test = torch.tensor(Y_real_test.values, dtype=torch.float32).view(-1, seq_len)
# X_train, X_test, y_train, y_test, y_real_train, y_real_test = train_test_split(train_tensor, ratings, True_ratings,
#                                                                                test_size=split_size, random_state=42)

print(f"\nX_train.shape: {X_train.shape}, X_test.shape: {X_test.shape},\n y_train.shape: {y_train.shape},  y_test.shape: {y_test.shape},\n y_real_train.shape: {y_real_train.shape}, y_real_test.shape: {y_real_test.shape}")
train_len = len(X_train)
test_len = len(X_test)
train_tensor = torch.tensor(X_train, dtype=torch.float32)
test_tensor = torch.tensor(X_test, dtype=torch.float32)
train_score_tensor = torch.tensor(y_train, dtype=torch.float32)
train_score_tensor = train_score_tensor.unsqueeze(dim=2)
test_score_tensor = torch.tensor(y_test, dtype=torch.float32)
test_score_tensor = test_score_tensor.unsqueeze(dim=2)
train_real_tensor = torch.tensor(y_real_train, dtype=torch.float32)
train_real_tensor = torch.unsqueeze(train_real_tensor, dim=2)
test_real_tensor = torch.tensor(y_real_test, dtype=torch.float32)
test_real_tensor = torch.unsqueeze(test_real_tensor, dim=2)

print(f"\ntrain_tensor.shape: {train_tensor.shape}, test_tensor.shape: {test_tensor.shape}\n"
      f"train_score_tensor.shape: {train_score_tensor.shape},test_score_tensor.shape: {test_score_tensor.shape}\n"
      f"train_real_tensor.shape: {train_real_tensor.shape}, test_real_tensor.shape: {test_real_tensor.shape}\n")

split_size = 0.4
# def preprocess_data():
    # Save train and test data to separate files
with open(f"train_tensor_{int((1 - split_size) * 100)}_{int(split_size * 100)}.pkl", 'wb') as f:
        pickle.dump(train_tensor, f)

with open(f"train_score_tensor_{int((1 - split_size) * 100)}_{int(split_size * 100)}.pkl", 'wb') as f:
        pickle.dump(train_score_tensor, f)

with open(f"train_real_tensor_{int((1 - split_size) * 100)}_{int(split_size * 100)}.pkl", 'wb') as f:
        pickle.dump(train_real_tensor, f)

with open("test_tensor.pkl", 'wb') as f:
        pickle.dump(test_tensor, f)

with open("test_score_tensor.pkl", 'wb') as f:
        pickle.dump(test_score_tensor, f)

with open("test_real_tensor.pkl", 'wb') as f:
        pickle.dump(test_real_tensor, f)

    

    # return train_tensor, train_score_tensor, train_real_tensor, test_tensor, test_score_tensor, test_real_tensor