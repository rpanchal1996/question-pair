import pandas as pd
file_path = '/home/rudresh/Desktop/quora/train.csv'
csv_dataframe  = pd.read_csv(file_path)
csv_dataframe = csv_dataframe[['question1','question2','is_duplicate']]
