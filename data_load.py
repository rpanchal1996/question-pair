import pandas as pd
file_path = '/home/rudresh/Desktop/quora/train.csv'
csv_dataframe  = pd.read_csv(file_path)
csv_dataframe = csv_dataframe[['question1','question2','is_duplicate']]
print csv_dataframe
len_avg = 0
number_of_items = 0
ctr = 0
for item in csv_dataframe['question1']:
	number_of_items+=1
	len_avg = len(str(item).strip().split()) + len_avg
	if len(str(item).strip().split()) >30:
		ctr+=1
print len_avg
print number_of_items
print str(ctr)
print str(len_avg/number_of_items)