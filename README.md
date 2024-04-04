In pre.sh there is a order in which code is to be run 
I have written separate codes for both test and training preprocessing.

1. python3 preprocess_train.py //  removing 0 relevance score from train file
2. python3 sample.py     // sampling 0.1 percent 
3. then we have svm commands to run
4. python3 ranking.py  : generates a csv file of query id and relevance score
5. python3 preprocss.py : genertaes a csv file for each query id having 10 entries: query_id,relevance_score,document_no,score
6. python3 seq_train.py   : creates a sequential data where each query id has its 10 documents in one row
7. echo "running for testing"
8.python3 ranking_test.py 
9.python3 preprocess_test.py 
10.python3 sequence.py
11.python3 features.py  // extract training features
12.python3 features_test.py   // extract test file features
13 .nohup python3 feature_set.py   // main feature extraction
14. nohup python3 main.py 
