import collections, itertools
import nltk.classify.util, nltk.metrics
import random
from sklearn import metrics 
from sklearn.svm import SVC
from nltk.classify import NaiveBayesClassifier , DecisionTreeClassifier , MaxentClassifier
from nltk.classify import SklearnClassifier
from nltk.corpus import movie_reviews, stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import scores

# k-fold definition
n_fold = 10

#function to calculate precision, recall and f1-measure value
#input: actual: A list of ground truth
#       pred: A list of your prediction
#output: precision , recall , f1-score
def calculate_result(actual , pred) :
	m_precision = metrics.precision_score(actual , pred)
	m_recall = metrics.recall_score(actual , pred)
	f1_score = metrics.f1_score(actual , pred)
	#print ("Precision:%1.3f Recall:%1.3f F1:%1.3f" % (m_precision , m_recall , f1_score) )
	return m_precision , m_recall , f1_score

#A function to print the final output in a certain format
def printing_result(accuracy , precision , recall , f1) :
	print ('accuracy:%1.3f' % accuracy)
	print ("Precision:%1.3f Recall:%1.3f F1:%1.3f" % (precision , recall , f1) )

#function to evaluate a feature
#input: A function which returns feature
#output: accuracy, precision , recall , f1_score
def evaluate_classifier(featx):
	negids = movie_reviews.fileids('neg')
	posids = movie_reviews.fileids('pos')

	negfeats = [(featx(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
	posfeats = [(featx(movie_reviews.words(fileids=[f])), 'pos') for f in posids]
	#Combine negative feature and positive feature and shuffle them
	totalfeats = negfeats + posfeats
	random.shuffle(totalfeats)
	#calculating subset_size for k-fold cross validation
	subset_size = int(len(totalfeats) / n_fold)
	
	#parameter initialization
	accuracy = 0.0
	precision = 0.0
	recall = 0.0
	f1 = 0.0
	
	#k_fold loop
	for j in range(n_fold) :
		testfeats = totalfeats[j*subset_size:][:subset_size]
		trainfeats = totalfeats[:j*subset_size]+ totalfeats[(j+1) * subset_size:]
		key = []
		result = []
		#classifier declaration : classifier can be swicthed here
		
		#classifier = NaiveBayesClassifier.train(trainfeats)
		classifier = SklearnClassifier(SVC(), sparse=False).train(trainfeats)
		
		#predicting
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
		print (j)
		p , r , f = calculate_result(key , result)
		precision += p
		recall += r
		f1 += f
	#classifier.show_most_informative_features() 
	return accuracy , precision , recall , f1
	
# baseline BOW feature extraction 
def word_feats(words):
    return dict([(word, True) for word in words])

# best score features extraction
#Get FreqDist and conditional freqdist
word_fd = FreqDist()
label_word_fd = ConditionalFreqDist()
 
for word in movie_reviews.words(categories=['pos']):
    word_fd.update([word.lower()])
    label_word_fd['pos'].update([word.lower()])
 
for word in movie_reviews.words(categories=['neg']):
    word_fd.update([word.lower()])
    label_word_fd['neg'].update([word.lower()])
 
# Get word count
pos_word_count = label_word_fd['pos'].N()
neg_word_count = label_word_fd['neg'].N()
total_word_count = pos_word_count + neg_word_count

#calculating word_scores
word_scores = {}
 
for word, freq in word_fd.items():
    pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word],
        (freq, pos_word_count), total_word_count)
    neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word],
        (freq, neg_word_count), total_word_count)
    word_scores[word] = pos_score + neg_score

# find the best word
best = sorted(word_scores.items(), key=lambda x : x[1], reverse=True)[:10000]
bestwords = set([w for w, s in best])
 
def best_word_feats(words):
    return dict([(word, True) for word in words if word in bestwords])
 
#best_bigram_word_feats
def best_bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    d = dict([(bigram, True) for bigram in bigrams])
    d.update(best_word_feats(words))
    return d
 
accuracy , precision , recall , f1 = evaluate_classifier(best_bigram_word_feats)
printing_result(accuracy/n_fold , precision/n_fold , recall/n_fold , f1/n_fold)
