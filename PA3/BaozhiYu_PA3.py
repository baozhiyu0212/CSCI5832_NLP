import re
import sys
import numpy as np
import argparse
import random
import math

#preprossessing the corpu in sentences
sentence = ''
sentence_n = []
sentence_test = ''
sentence_n_test = []

#function to split the training data into 80% training data and 20% development data
def traindevsplit(file_object , ratio) :
	file_object = open(file_object,'r')
	index = [i for i in range(15370)]
	random.shuffle(index)
	seed_dev = [index[i] for i in range(round(15370*ratio))]
	train = open('train.txt','w')
	dev = open('dev.txt','w')
	count = 0
	for line in file_object.readlines() :
		if count not in seed_dev :
			train.writelines(line)
		else :
			dev.writelines(line)
		if len(line) == 1 :
			count += 1
	return train, dev

#function to get tag count
def GetTagCount(file) :
	cnt_tag = {}
	vol = {}
	num = 0
	for line in file.readlines() :
		#line = line.lower()
		num += 1
		split = line.split()
		if split:
			if split[0] not in vol :
				vol[split[0]] = 1
			else :
				vol[split[0]] += 1
			if split[1] not in cnt_tag :
				cnt_tag[split[1]] = 1
			else :
				cnt_tag[split[1]] += 1
	cnt_tag['<s>'] = num
	vol['<s>'] = num
	return cnt_tag , vol

# function to get word-pair count
def GetWordtagCount(file) :
	cnt_wordtag = {}
	for line in file.readlines() :
		line = line.split()
		if line :
			line = line[0] + ' ' + line[1]
			if line not in cnt_wordtag :
				
				cnt_wordtag[line] = 1
			else :
				#line = line.split[0] + ' ' + line.split[1]
				cnt_wordtag[line] += 1
	return cnt_wordtag

#function to get Tagbigram count	
def GetTagBigramCount(file) :
	tag_train = ''
	tag_bi = []
	bigram = {}
	count = 0
	for line in file.readlines() :
		split = line.split()
		if len(split) == 2 :
			tag_train += split[1] + ' '
	tag_train = re.split('[.]' , tag_train)
	for word in tag_train :
		word = '<s>' + word + '.'
		tag_bi.append(word)
	for word in tag_bi :
		word = word.split()
		stri = ''
		for i in range(len(word)-1) :
			stri = word[i] + " " + word[i+1]
			if stri not in bigram :
				bigram[stri] = 1
			else :
				bigram[stri] += 1
	return bigram

#Compute Observation matrix(dictionary)
def ComputeOLikelihood(tag , wordtag , vol) :
	O = {}
	for t in tag:
		if t not in O :
			O[t] = {}
			for word in vol :
				if word not in O[t] :
					O[t][word] = 0.0000000001
	for word in wordtag :
		split = word.split()
		one = split[0]
		two = split[1]
		if len(split) == 2 :
			numerator = (wordtag[word])
			denominator = (tag[split[1]])
			O[two][one] = numerator / denominator
	#print (O)
	return O

#Compute Transition matrix(dictionary)
def ComputeTlikelihood(tag , tag_bigram) :
	T = {}
	for i in tag :
		if i not in T :
			T[i] = {}
			for j in tag :
				if j not in T[i] :
					T[i][j] = 0.0000000001
	for i in tag_bigram :	
		one = i.split()[0]
		two = i.split()[1]
		if tag.get(one) != None :
			T[one][two] = tag_bigram.get(i) / tag.get(one)
	#print (T)
	return T
	

#Viterbi algorithm
#obs : observation sequence
#states : state sequence
#start_p : prior probability
#trans_p : Transition matrix
#emit_p : Observation matrix
def viterbi(obs, states, start_p, trans_p, emit_p):
	T = {}
	#for the first one(initialization step)
	for state in states:
		T[state] = (start_p[state] * emit_p[state][obs[0]], [state], start_p[state])
	for output in obs:
		U = {}
		#pass the first one
		if output == obs[0]:
			continue
		for next_state in states:
			total = 0
			argmax = None
			valmax = 0
			for source_state in states:
				#get the previous one
				(prob, v_path, v_prob) = T[source_state]
				#caculate the commone one
				if output not in vol :
					p = trans_p[source_state][next_state] * 0.000000001
				else :
					p = trans_p[source_state][next_state] * emit_p[next_state][output]
				prob *= p
				v_prob *= p
				#sum up
				total += prob
				#find the max
				if v_prob > valmax:
					argmax = v_path + [next_state]
					valmax = v_prob
			U[next_state] = (total, argmax, valmax)
		T = U
	#sum up & find the max one
	total = 0
	argmax = None
	valmax = 0
	for state in states:
		(prob, v_path, v_prob) = T[state]
		total += prob
		if v_prob > valmax:
			valmax = v_prob
			argmax = v_path
	return argmax

	
				
				

if __name__ == "__main__":
	# split train and dev dataset
	train , dev = traindevsplit(sys.argv[1] , 0.2)
	#split training data model initialization
	train = open("train.txt",'r')
	tag , vol = GetTagCount(train)
	train = open("train.txt",'r')
	wordtag = GetWordtagCount(train)
	#Observation dictionary calculation
	Observation = ComputeOLikelihood(tag , wordtag , vol)
	train = open("train.txt",'r')
	tag_bigram = GetTagBigramCount(train)
	#Transition dictionary calculation
	Transition = ComputeTlikelihood(tag , tag_bigram)
	state_graph = tuple(tag.keys())
	start_probability = {}
	for i in state_graph :
		start_probability[i] = 1/len(state_graph)
	
	#validation part	
	dev = open("dev.txt",'r')
	for line in dev.readlines() :
		split = line.split()
		if len(split) == 2 :
			sentence += split[0] + ' '
	sentence = re.split('\s\.\s' , sentence)
	for word in sentence :
		word = '<s> ' + word + ' .'
		sentence_n.append(word)
	key = open('validation_key.txt','w')
	for i in range(len(sentence_n)-1):
		target = tuple(sentence_n[i].split())
		result = viterbi(target , state_graph , start_probability , Transition , Observation)
		result.remove(result[0])
		target = list(target)
		target.remove(target[0])
		for i in range(len(target)) :
			key.writelines(target[i] + '\t' + result[i] + '\n')
		key.writelines('\n')
	key.close()

	
	
	# all training data model initialization
	train_test = open(sys.argv[1],'r')
	tag_test , vol_test = GetTagCount(train_test)
	train_test = open(sys.argv[1],'r')
	wordtag_test = GetWordtagCount(train_test)
	# Observation dictionary calculation
	Observation_test = ComputeOLikelihood(tag_test , wordtag_test , vol_test)
	train_test = open(sys.argv[1],'r')
	tag_bigram_test = GetTagBigramCount(train_test)
	# Transition dictionary calculation
	Transition_test = ComputeTlikelihood(tag_test , tag_bigram_test)	
	state_graph_test = tuple(tag_test.keys())
	start_probability_test = {}
	for i in state_graph_test :
		start_probability_test[i] = 1/len(state_graph_test)
	# test part	
	test = open(sys.argv[2],'r')
	for line in test.readlines() :
		if len(line) != 1:
			split_test = line.split()
			sentence_test += split_test[0] + ' '
	sentence_test = re.split('\s\.\s' , sentence_test)
	for word in sentence_test :
		word = '<s> ' + word + ' .'
		sentence_n_test.append(word)
	key_test = open('test_key.txt','w')
	num = 0
	for i in range(len(sentence_n_test)-1):
		target_test = tuple(sentence_n_test[i].split())
		result_test = viterbi(target_test , state_graph_test , start_probability_test , Transition_test , Observation_test)
		result_test.remove(result_test[0])
		target_test = list(target_test)
		target_test.remove(target_test[0])
		num += 1
		for i in range(len(target_test)) :
			key_test.writelines(target_test[i] + '\t' + result_test[i] + '\n')
			print(target_test[i] + '\t' + result_test[i])
		#hard-coded part for wrong format in test file
		if num != 1636 :
			key_test.writelines('\n')
			print (' ')
		if num == 1608 :
			key_test.writelines('\n')
			print (' ')
	key_test.close()

	
	
	
	
	
	
	
	
	
	
	
