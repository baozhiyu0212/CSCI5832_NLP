import collections, itertools
import nltk.classify.util, nltk.metrics
import random

from sklearn import metrics 
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews, stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import scores

n_fold = 10

def calculate_result(actual , pred) :
	m_precision = metrics.precision_score(actual , pred)
	m_recall = metrics.recall_score(actual , pred)
	f1_score = metrics.f1_score(actual , pred)
	#print ("Precision:%1.3f Recall:%1.3f F1:%1.3f" % (m_precision , m_recall , f1_score) )
	return m_precision , m_recall , f1_score

def printing_result(accuracy , precision , recall , f1) :
	print ('accuracy:%1.3f' % accuracy)
	print ("Precision:%1.3f Recall:%1.3f F1:%1.3f" % (precision , recall , f1) )

def evaluate_classifier(featx):
	negids = movie_reviews.fileids('neg')
	posids = movie_reviews.fileids('pos')

	negfeats = [(featx(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
	posfeats = [(featx(movie_reviews.words(fileids=[f])), 'pos') for f in posids]
 
	totalfeats = negfeats + posfeats
	random.shuffle(totalfeats)
	subset_size = int(len(totalfeats) / n_fold)
	
	accuracy = 0.0
	precision = 0.0
	recall = 0.0
	f1 = 0.0
	
	for j in range(n_fold) :
		testfeats = totalfeats[j*subset_size:][:subset_size]
		trainfeats = totalfeats[:j*subset_size]+ totalfeats[(j+1) * subset_size:]
		classifier = NaiveBayesClassifier.train(trainfeats)
		key = []
		result = []

		for i, (feats, label) in enumerate(testfeats):
			if label == 'pos' :
				key.append(1)
			else :
				key.append(0)
			observed = classifier.classify(feats)
			if observed == 'pos' :
				result.append(1)
			else :
				result.append(0)
		
		accuracy += nltk.classify.util.accuracy(classifier, testfeats)
		p , r , f = calculate_result(key , result)
		precision += p
		recall += r
		f1 += f
	classifier.show_most_informative_features(20000) 
	return accuracy , precision , recall , f1
	
# baseline BOW feature extraction 
def word_feats(words):
    return dict([(word, True) for word in words])
 
print ('evaluating BOW features')	
accuracy , precision , recall , f1 = evaluate_classifier(word_feats)
printing_result(accuracy/n_fold , precision/n_fold , recall/n_fold , f1/n_fold)


# best score features extraction

word_fd = FreqDist()
label_word_fd = ConditionalFreqDist()
 
for word in movie_reviews.words(categories=['pos']):
    word_fd.update([word.lower()])
    label_word_fd['pos'].update([word.lower()])
 
for word in movie_reviews.words(categories=['neg']):
    word_fd.update([word.lower()])
    label_word_fd['neg'].update([word.lower()])
 
 
pos_word_count = label_word_fd['pos'].N()
neg_word_count = label_word_fd['neg'].N()
total_word_count = pos_word_count + neg_word_count
 
word_scores = {}
 
for word, freq in word_fd.items():
    pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word],
        (freq, pos_word_count), total_word_count)
    neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word],
        (freq, neg_word_count), total_word_count)
    word_scores[word] = pos_score + neg_score

best = sorted(word_scores.items(), key=lambda x : x[1], reverse=True)[:10000]
bestwords = set([w for w, s in best])
 
def best_word_feats(words):
    return dict([(word, True) for word in words if word in bestwords])
 
#print ('evaluating best word features')
#accuracy , precision , recall , f1 = evaluate_classifier(best_word_feats)
#printing_result(accuracy/n_fold , precision/n_fold , recall/n_fold , f1/n_fold)
