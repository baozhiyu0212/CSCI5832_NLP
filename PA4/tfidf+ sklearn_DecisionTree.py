from nltk.corpus import movie_reviews
import random
import numpy as np
import nltk
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer , TfidfTransformer
from sklearn import metrics 
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

def calculate_result(actual , pred) :
	m_precision = metrics.precision_score(actual , pred)
	m_recall = metrics.recall_score(actual , pred)
	f1_score = metrics.f1_score(actual , pred)
	#print ("Precision:%1.3f Recall:%1.3f F1:%1.3f" % (m_precision , m_recall , f1_score) )
	return m_precision , m_recall , f1_score

def printing_result(accuracy , precision , recall , f1) :
	print ('accuracy:%1.3f' % accuracy)
	print ("Precision:%1.3f Recall:%1.3f F1:%1.3f" % (precision , recall , f1) )

n_fold = 10
	
if __name__ == "__main__":
	
	documents = [(list(movie_reviews.words(fileid)) , category) 
				for category in movie_reviews.categories()
				for fileid in movie_reviews.fileids(category)]			
	random.shuffle(documents)
	
	all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
	word_features = list(all_words)[:2000]
	
	accuracy = 0
	precision = 0
	recall = 0
	f1_score = 0

	for j in range(n_fold) :
		seed = [i for i in range(len(documents))]
		seed = seed[int(len(documents)*j/10 ) : int(len(documents)*(j+1)/10)]
		train_x = []
		train_y = []
		test_x = []
		test_y = []
		for i in range(len(documents)) :
			if i not in seed :
				train_x.append(str(documents[i][0]))
				train_y.append(documents[i][1])
			else :
				test_x.append(str(documents[i][0]))
				test_y.append(documents[i][1])
		
		count_v1 = CountVectorizer(vocabulary = word_features)
		count_train = count_v1.fit_transform(train_x)

		count_v2 = CountVectorizer(vocabulary = count_v1.vocabulary_)
		count_test = count_v2.fit_transform(test_x)
		
		tfidf = TfidfTransformer()
		
		tfidf_train = tfidf.fit(count_train).transform(count_train)
		tfidf_test = tfidf.fit(count_test).transform(count_test)
		
		#clf = MultinomialNB()
		clf =  DecisionTreeClassifier(max_depth = 3)
		#clf = SVC()	
		clf.fit(tfidf_train , train_y)
		accuracy += clf.score(tfidf_test , test_y)
		#print ("Accuracy:%f" % clf.score(tfidf_test , test_y))
		pred = clf.predict(tfidf_test)
		pred = list(pred)
		for i in range(len(test_y)) :
			if test_y[i] == 'pos' :
				test_y[i] = 1
			else:
				test_y[i] = 0
		for i in range(len(pred)) :
			if pred[i] == 'pos' :
				pred[i] = 1
			else:
				pred[i] = 0
		p , r , f = calculate_result(test_y , list(pred))
		
		precision += p
		recall += r
		f1_score += f
	
printing_result(accuracy/n_fold , precision/n_fold , recall/n_fold , f1_score/n_fold)



