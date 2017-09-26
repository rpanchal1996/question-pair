import pandas as pd
from nltk.stem.porter import PorterStemmer
import re
import numpy as np
import json
maxSeqLength = 30
ps = PorterStemmer()
def load_data_saved():
	with open('stemmed_split_sentences','r') as myfile:
		data = json.load(myfile)
	return data

def load_data():
	file_path = '/home/rudresh/Desktop/quora/train.csv'
	csv_dataframe  = pd.read_csv(file_path)
	csv_dataframe = csv_dataframe[['question1','question2','is_duplicate']]
	question1 = []
	question2 = []
	is_duplicate = []
	for index, row in csv_dataframe.iterrows():
		q1 = str(row['question1'])
		q2 = str(row['question2'])
		question1_cleaned = clean_text(q1.lower())
		question2_cleaned = clean_text(q2.lower())
		question1_words = question1_cleaned.split()
		question2_words = question2_cleaned.split()
		question1_words = [ps.stem(word) for word in question1_words]
		question2_words = [ps.stem(word) for word in question2_words]
		if len(question1_words)>30 or len(question2_words)>30:
			pass
		else:
			question1.append(question1_words)
			question2.append(question2_words)
			is_duplicate.append(row['is_duplicate'])
	zipped_object = zip(question1,question2,is_duplicate)
	with open('stemmed_split_sentences','w') as myfile:
		json.dump(zipped_object,myfile)
	
	return zip(question1,question2,is_duplicate)
def clean_text(text):
	text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
	text = re.sub(r"what's", "what is ", text)
	text = re.sub(r"\'s", " ", text)
	text = re.sub(r"\'ve", " have ", text)
	text = re.sub(r"can't", "cannot ", text)
	text = re.sub(r"n't", " not ", text)
	text = re.sub(r"i'm", "i am ", text)
	text = re.sub(r"\'re", " are ", text)
	text = re.sub(r"\'d", " would ", text)
	text = re.sub(r"\'ll", " will ", text)
	text = re.sub(r",", " ", text)
	text = re.sub(r"\.", " ", text)
	text = re.sub(r"!", " ! ", text)
	text = re.sub(r"\/", " ", text)
	text = re.sub(r"\^", " ^ ", text)
	text = re.sub(r"\+", " + ", text)
	text = re.sub(r"\-", " - ", text)
	text = re.sub(r"\=", " = ", text)
	text = re.sub(r"'", " ", text)
	text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
	text = re.sub(r":", " : ", text)
	text = re.sub(r" e g ", " eg ", text)
	text = re.sub(r" b g ", " bg ", text)
	text = re.sub(r" u s ", " american ", text)
	text = re.sub(r"\0s", "0", text)
	text = re.sub(r" 9 11 ", "911", text)
	text = re.sub(r"e - mail", "email", text)
	text = re.sub(r"j k", "jk", text)
	text = re.sub(r"\s{2,}", " ", text)
	return text

def vectorize():
	wordlist = []
	known = 0
	unkown = 0
	with open('wordlist','r') as myfile:
		wordlist = myfile.readlines()
		wordlist = [word.lower().strip() for word in wordlist]
	zipped_data = load_data_saved()
	number_of_examples = len(zipped_data)	
	question_one_ids = np.zeros((number_of_examples, maxSeqLength), dtype='int32')
	question_two_ids = np.zeros((number_of_examples, maxSeqLength), dtype='int32')
	is_duplicate = np.zeros((number_of_examples,1),dtype = 'int32')
	example_counter = 0
	
	for question1_words,question2_words,is_duplicate in zipped_data:
		wordcounter = 0
		print question1_words
		print question2_words
		print example_counter
		for word in question1_words:
			try:
				question_one_ids[example_counter][wordcounter] = wordlist.index(word)
				known+=1
				wordcounter+=1
			except ValueError:								     
				question_one_ids[example_counter][wordcounter] = 3999999 #Vector for unkown words
				wordcounter+=1
				unkown+=1
		wordcounter = 0
		for word in question2_words:
			try:
				question_two_ids[example_counter][wordcounter] = wordlist.index(word)
				known+=1
				wordcounter+=1
			except ValueError:								     
				question_two_ids[example_counter][wordcounter] = 3999999 #Vector for unkown words
				wordcounter+=1
				unkown+=1
		example_counter+=1
		if example_counter % 10 == 0:
			print 'NUMBER OF EXAMPLES DONE === ' + str(example_counter)
			print 'TOTAL NUMBER OF SHIT    === ' + str(number_of_examples) 
		wordcounter = 0
	np.save('q1_ids_matrix',question_one_ids)
	np.save('q2_ids_matrix',question_two_ids)	
	print known
	print unkown

def check_saved_id_matrix():
	zipped_data = load_data()
	for question1,question2,is_duplicate in zipped_data[1:2]:
		print question1
		print question2
	question_one_ids = np.load('q1.npy')
	question_two_ids = np.load('q2.npy')
	print question_one_ids[0]
	print question_two_ids[0]

vectorize()
