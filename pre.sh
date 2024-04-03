!/bin/bash

echo "running for training"

python3 preprocess_train.py
# python3 test_preprocess.py
python3 sample.py
./svm_rank_learn -c 200 train_sampled.txt prod_rank.model 
./svm_rank_classify train_filtered.txt prod_rank.model predictions_train.txt 
./svm_rank_classify set1.test.txt prod_rank.model predictions_test.txt 
echo "svm done"
python3 ranking.py 
python3 preprocess.py 
python3 seq_train.py 
echo "running for testing"
python3 ranking_test.py 
python3 preprocess_test.py 
python3 sequence.py 
python3 features.py 
python3 features_test.py 
nohup python3 feature_set.py 
nohup python3 main.py