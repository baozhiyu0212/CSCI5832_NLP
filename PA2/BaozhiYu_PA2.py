import sys
import math


#initial vocabulary
v_uni = {}
v_bi = {}
prob_uni = {}
prob_bi = {}
prob_smooth = {}

#Get Unigram Count
def GetUnigramCounts(file) :
	num_train = 0
	for line in file.readlines() :
		line = line.lower()
		line = line.split()
		for word in line :
			num_train += 1
			if word not in v_uni :
				v_uni[word] = 1
			else :
				v_uni[word] += 1 
	file.close()
	return v_uni, num_train

#Get Bigram Count
def GetBigramCounts(file) :
	for line in file.readlines() :
		line = "%s <\s>" % line
		line = line.lower()
		line = line.split()
		num = 0
		stri = ''
		for i in range(len(line)-1) :
			stri = line[i] + " " + line[i+1]
			if stri not in v_bi :
				v_bi[stri] = 1
			else :
				v_bi[stri] += 1
	file.close()
	return v_bi
	
#Get Unigram probability
def GetUnigramprobabilities(unigram , num) :
	for i in unigram :
		prob_uni[i] = unigram[i] / num
	return prob_uni

#Get Bigram probability
def GetBigramProbabilities(bigram , probunigram) :
	for word in bigram :
		numerator = bigram[word]
		pre = word.split()[0]
		denominator = unigram[pre]
		prob_bi[word] = numerator / denominator
	return prob_bi

#Get Add-k smoothed probability
def GetSmoothedProbabilities(bigram , k , v) :
	for word in bigram :
		numerator = bigram[word] + k
		pre = word.split()[0]
		denominator = unigram[pre] + v * k
		prob_smooth[word] = numerator / denominator
	return prob_smooth

#Get Unigram Sent Probability
def GetUnigramSentProbability(line , probunigram) :
	prob = 0
	for word in line :
		prob = math.log10(probunigram[word]) + prob
	return prob

#Get Bigram Sent Probability
def GetBigramSentProbability(line , probbigram) :
	prob = 0
	stri = ''
	for i in range(len(line)-1) :
		stri = line[i] + " " + line[i+1]
		if stri not in probbigram :
			#print (stri)
			return 0
		else :
			prob = prob + math.log10((probbigram[stri]))
	return prob

#Get Smoothed Sent Probability
def GetSmoothedSentProbability(line , probsmooth) :	
	prob = 0
	stri = ''
	for i in range(len(line)-1) :
		stri = line[i] + " " + line[i+1]
		if stri not in probsmooth :
			prob = prob + math.log10((0 + 0.0001) / (unigram[line[i]] + 0.0001 * v))
		else :
			prob = math.log10(probsmooth[stri]) + prob
	return prob

if __name__ == "__main__":
	file_train = open(sys.argv[1],'r')
	file_test = open(sys.argv[2],'r')
	unigram , total_token = GetUnigramCounts(file_train)
	uniprob = GetUnigramprobabilities(unigram , total_token)
	
	file_train = open(sys.argv[1],'r')
	bigram = GetBigramCounts(file_train)
	biprob = GetBigramProbabilities(bigram , uniprob)
	
	v = len(unigram)
	smoothprob = GetSmoothedProbabilities(bigram , 0.0001 , v)
	
	for line in file_test.readlines() :
		print ("Sentence = %s" % line)
		line = line.lower()
		line = line.split()
		prob_unigram = GetUnigramSentProbability(line , uniprob)
		print ("Unigrams:logprob(S) = %.4f" % prob_unigram)
		prob_bigram = GetBigramSentProbability(line , biprob)
		if prob_bigram == 0 :
			print ("Bigrams:logprob(S) = 1.0000")
		else :
			print  ("Bigrams:logprob(S) = %.4f" % prob_bigram)
		prob_smoothed = GetSmoothedSentProbability(line , smoothprob)
		print ("Smoothed Bigrams:logprob(S) = %.4f" % prob_smoothed)
	
				
	file_test.close()
	
