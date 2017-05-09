import re
import sys
import numpy as np
import argparse

#paragraph counting function
def paragraph_count() :
	file_object = open(sys.argv[1],'r')
	paragraph = 1
	for line in file_object.readlines() :
		if len(line) == 1 :
			paragraph += 1
	file_object.close()
	return paragraph

#sentence counting function
def sentence_count() :
	file_object = open(sys.argv[1],'r')
	sentence = 0
	for line in file_object.readlines() :
		if len(line) != 1 :
			#pattern = re.compile(r'.*[\?\!]|.*\.\"|.*\.\) [A-Z]|.*\.\r\n')
			pattern1 = re.compile(r'[\w"]\.\s+[A-Z"]|[\?\!]\s+|\.\"\s+|\)\.\s+')
			pattern4 = re.compile(r'Dr\.\s+[A-Z]|Corp\.\s+[A-Z]|A\.\s+[A-Z]|T\.\s+[A-Z]|Mr\.\s+[A-Z]|Ms\.\s+[A-Z]')
			sentences1 = pattern1.split(line)
			sentences4 = pattern4.split(line)
			
			#print (line)
			#print (sentences1)
			#print (len(sentences4))
			count = (len(sentences1)-1-(len(sentences4)-1) + 1)
			sentence += count
	file_object.close()
	return sentence+1

#word counting function
def word_count() :
	file_object = open(sys.argv[1],'r')
	word = 0
	for line in file_object.readlines() :
		words = line.split(' ')
		#print line
		if '\n' in words :
			words.remove('\n')
		if '"' in words :
			words.remove('"')
		if '-' in words :
			words.remove('-')
		if '--' in words :
			words.remove('--')
		if '&' in words :
			words.remove('&')
		#print words
		word += len(words)
		#print len(words)
	file_object.close()
	return word

#result printing function	
def print_result(paragraph,sentence,word) :
	print ("Paragraph: %d" % paragraph)
	print ("Sentence: %d" % sentence)
	print ("Word: %d" % word)


if __name__ == "__main__":

	paragraph_number = paragraph_count()
	sentence_number = sentence_count()
	word_number = word_count()
	print_result(paragraph_number,sentence_number,word_number)
	
