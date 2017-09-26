import pandas as pd
file_path = '/home/rudresh/Desktop/quora/train.csv'
csv_dataframe  = pd.read_csv(file_path)
csv_dataframe = csv_dataframe[['question1','question2','is_duplicate']]
question1 = []
question2 = []
is_duplicate = []
for index, row in csv_dataframe.iterrows():
	q1 = str(row['question1'])
	q2 = str(row['question2'])
	if len(q1.split())>30 or len(q2.split())>30:
		pass
	else:
		question1.append(q1.lower())
		question2.append(q2.lower())
		is_duplicate.append(row['is_duplicate'])
print len(question1)
print len(question2)
