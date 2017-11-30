import sklearn
import numpy as np
import re

from scipy.stats import ttest_ind
from sklearn.feature_extraction.text import TfidfVectorizer

print("Sklearn version: "+sklearn.__version__)

def read_dataset(file_path):
	data = []
	f = open(file_path, 'r')
	for oneline in f.readlines():
		split = oneline.strip("\n").split("\t")
		mark = 0 if (split[0]=="ham") else 1
		data.append((mark, split[1]))
		#print(str(mark)+"->"+split[1])
	f.close()
	return data


mark2data_pair = read_dataset("datasets/SMSSpamCollection")

def spams_count(pair_list):
	count = 0
	for (mark, txt) in pair_list:
		count += 1 if mark == 1 else 0
	
	return count

print("Spam count="+str(spams_count(mark2data_pair)))

#zipped_itr = list(zip(*mark2data_pair))[0]
#print("Spam count by zip="+str(sum(zipped_itr)))
tfidf_model = TfidfVectorizer(min_df=0.001, stop_words = "english")
def transform_text(pair_list):
	x_raw = []
	y_raw = []
	for (mark, txt) in pair_list:
		x_raw.append(txt)
		y_raw.append(mark)
	#for v in tfidf_model.get_stop_words():
	#	print(v)
	print("==============")
	X = tfidf_model.fit_transform(x_raw)
	vocabs = tfidf_model.get_feature_names()
	# Write to file
	f = open("vocanularies.txt", "w")
	count = 0
	for v in vocabs:
		count+=1
		f.write(v+"\t")
		if(count%5==0):
			f.write("\n")
	f.close()
	return (X, np.array(y_raw))

# [letter for letter in text if letter not in string.punctuation]
# " ".join(text)

# Pour 1 mot :
# Comparer les valeurs de ce mot pon spam et non-spam
# La difference des moyennes est-elle 0?
def test_words_means(x, y, word_index) :
	X_spam = x[y==1, word_index].todense()
	X_ham = x[y==0, word_index].todense()
	return ttest_ind(X_spam, X_ham)[1]

p_values = []
X, Y = transform_text(mark2data_pair)
vocabulary = tfidf_model.get_feature_names()
for word_index, word in enumerate(vocabulary):
	p_value= test_words_means(X, Y, word_index)
	print("Means for '"+word+"' = "+str(p_value))
	p_values.append((word, p_value))

p_values.sort(key=lambda x: x[1])

print(p_values)
#def print_discriminative_words(p_values, threadhold)

