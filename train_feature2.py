import pandas as pd 
from tqdm import tqdm
def get_tensor2():
    df = pd.read_csv('final_output_test.csv')
    df1 = pd.read_csv('features_test.csv')
    X = pd.DataFrame()
    column_names1 = ['score1' , 'score2' , 'score3' ,'score4' , 'score5' , 'score6' , 'score7' , 'score8' , 'score9' , 'score10']
    Y_train = pd.DataFrame(columns = column_names1)
    column_names2 = ['real_score1' , 'real_score2' , 'real_score3' ,'real_score4' , 'real_score5' , 'real_score6' , 'real_score7' , 'real_score8' , 'real_score9' , 'real_score10']
    Y_real = pd.DataFrame(columns = column_names2)
    for index, entry in tqdm(df.iterrows(), total=df.shape[0]):
        query_id = entry['query_id']

        doc_columns =  [col for col in df.columns if "doc" in col]
        ind = 1 
        y=[]
        y_real = []
        for document in doc_columns:
            number = entry[document]
            click_probability = entry[f'click_prob_{ind}']
            true_score = entry[f'relevance_score_{ind}']
            filtered_df = df1[(df1['query_id'] == query_id) & (df1['document_number'] == number)]
            features = filtered_df.filter(regex='^feature').copy()
            features = features.drop(columns=['feature3111'], errors='ignore')
            features['query_id'] = query_id
            X = pd.concat([X, features], ignore_index=True)
            y_real.append(true_score)
            y.append(click_probability)
            ind+=1
        new_row = pd.DataFrame([y], columns=Y_train.columns)
        Y_train = pd.concat([Y_train, new_row], ignore_index=True)

        new_row2 = pd.DataFrame([y_real], columns=Y_real.columns)
        Y_real = pd.concat([Y_real, new_row2], ignore_index=True)
        
        

    return X , Y_train , Y_real

        








