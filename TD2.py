import sklearn
import numpy as np
import re

from scipy.stats import ttest_ind
from sklearn.feature_extraction.text import TfidfVectorizer

print(sklearn.__version__)

data = []
f = open('SMSSpamCollection', 'r')
for oneline in f.readlines():
	split = oneline.strip("\n").split("\t")
	mark = 0 if (split[0]=="ham") else 1
	data.append((mark, split[1]))
	print(str(mark)+"->"+split[1])


def spams_count(pair_list):
	count = 0
	for (mark, txt) in pair_list:
		count += 1 if mark == 1 else 0
	
	return count

print(spams_count(data))

zipped_itr = list(zip(*data))[0]
print(sum(zipped_itr))

def transform_text(pair_list):
	x = []
	y = []
	for (mark, txt) in pair_list:
		txt_lower_case = txt.lower()
		
		x.append()
		y.append(mark)

# [letter for letter in text if letter not in string.punctuation]
# " ".join(text)


tfidf_model = TfidfVectorizer(min_df=0.1)
tfidf = tfidf_model.fit_transform(X)



