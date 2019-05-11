import requests
import pipeline_caller
import json
import os
import argparse
import re
import pickle
import math
import numpy as np
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC

TOKENFILE_PATH = 'pipeline.token'

parser = argparse.ArgumentParser()
parser.add_argument('-mode', action='store',default='read',help='The mode of the program. Usage : *read to read sentence files *convert to go through existing files')
parser.add_argument('-readfile', action='store', help='Choose the file that contains the sentences to be read.')
parser.add_argument('-append', action='store_true', help='Creates additional files in the folder instead of writing over them. (Read mode only)')


def isInteger(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

SENTENCEOFFSET = 3
SKIPWORD = "<S>"
KEYWORDS = ["daha", "en", "gibi", "kadar","göre", "hem"]
KEYSUFFIX = 'Abl'

if __name__ == "__main__":
	args = parser.parse_args()
	addfiles = args.append
	readfile = args.readfile
	mode = args.mode

	curpath = os.getcwd()

	writedir_api = curpath + "/apiresults/train/"
	writedir_api_test = curpath + "/apiresults/test/"
	writedir_convert = curpath + "/convertresults/"

	if not (os.path.exists(writedir_api)):
		os.makedirs(writedir_api)
	if not (os.path.exists(writedir_api_test)):
		os.makedirs(writedir_api_test)
	if not (os.path.exists(writedir_convert)):
		os.makedirs(writedir_convert)

	if (args.mode == "read"):
		fileoffset = 0

		with open(TOKENFILE_PATH,'r',) as tokenfile:
			token = tokenfile.read().strip() #get token here
			tokenfile.close()

		caller = pipeline_caller.PipelineCaller()

		sentencelist = []

		if (addfiles):
			fileoffset = len(os.listdir(writedir_api))

		if (readfile):
			readdir = os.getcwd() + "/" + readfile
			with open(readdir,'r', encoding='utf-8') as r:
				sentencelist = r.readlines()
			r.close()

		for index,sent in enumerate(sentencelist):
			infolist = []
			splitdata = [x.strip() for x in sent.split('|')]
			#actual sentence
			sentmain = splitdata[0]
			#whether it's comparative or not
			sentclass = splitdata[1]
			print(splitdata)
			res = caller.call(tool='pipelineFormal',text=sentmain, token=token)

			for line in res.splitlines():		
				worddict = {}
				parameter_list = ["id","word","root","pos1","pos2","nounsuffixes","dependency","relation"]
				linedata = line.split()
				for idx,data in enumerate(linedata):
					if (isInteger(data)):
						data = int(data)
					worddict[parameter_list[idx]] = data
				infolist.append(worddict)

			infolist.append({"class" : sentclass})
			with open(writedir_api + "results_" + str(index + fileoffset) + ".txt", "w", encoding="utf-8") as f:
				#dump the dictionary
				f.writelines(json.dumps(l) + "\n" for l in infolist)
				f.close()
	elif (args.mode == "convert"):

		#read the api files from this folder
		apifiles = [os.path.join(writedir_api,f) for f in os.listdir(writedir_api)]

		#add extracted sequences here
		features = []

		for file in apifiles:
			with open(file) as f:

				jsondata = [json.loads(line) for line in f.readlines()]
				#get class
				sentenceclass = jsondata.pop()
				#remove derivated words, they aren't part of the actual sentence

				jsondata_filter = [word for word in jsondata if (word["relation"] != "DERIV" and word["relation"] != "PUNCTUATION")]
				for idx,elem in enumerate(jsondata_filter):
					#check if the word is a keyword, or contains -dan,-den
					hasKeyword = (elem["word"] in KEYWORDS) or (elem["root"] in KEYWORDS)
					hasSuffix = re.search(KEYSUFFIX, elem["nounsuffixes"])

					print(idx)
					if (hasKeyword or hasSuffix):
						min_elem = max(idx-SENTENCEOFFSET,0)
						max_elem = min(idx+SENTENCEOFFSET,len(jsondata_filter)-1)+1

						backappend = (SENTENCEOFFSET-idx) + min_elem
						frontappend = (idx + SENTENCEOFFSET) - max_elem + 1 

						#change the POS with a special POS
						if (hasKeyword):
							elem["pos1"] = elem["word"] + elem["pos1"]
						elif (hasSuffix):
							elem["pos1"] = KEYSUFFIX + elem["pos1"]

						sequence = [chosen_elem["pos1"] for chosen_elem in jsondata_filter[min_elem:max_elem]]

						#append default POS tags until the sequence sizes are equal
						for i in range(0,backappend):
							sequence.insert(0,"<S>")
						for i in range(0,frontappend):
							sequence.append("<S>")

						feature1 = {"sequence" : sequence, "class": sentenceclass["class"]}

						
						print(elem["word"])
						print(feature1)

						features.append(feature1)


		with open(writedir_convert + "featurelist.txt", "w", encoding="utf-8") as f:
			f.writelines((json.dumps(feat) + "\n") for feat in features)
	elif (args.mode == "train"):
		word_id_list = {}
		docsize = 0
		
		with open(writedir_convert + "featurelist.txt", "r", encoding="utf-8") as f:
			data_read = [json.loads(data) for data in f.readlines()]
			for data in data_read:
				docsize += 1
				for word in word_id_list.keys():
					word_id_list[word]['hasappeared'] = False
				numseq = []
				dseq = data["sequence"]
				dclass = data["class"]
				for word in dseq:
					if word == SKIPWORD:
						pass
					else:
						if word in word_id_list.keys():
							word_id_list[word]['rawcount'] += 1
							if (not word_id_list[word]['hasappeared']):
								word_id_list[word]['doccount'] += 1
								word_id_list[word]['hasappeared'] = True
						else:						
							word_id_list[word] = {
							'tfidf' : 0.0, 
							'inverse_frequency' : 0.0, 
							'raw_frequency' : 0.0, 
							'rawcount' : 1,
							'doccount' : 1,
							'hasappeared' : True
							}

			f.close()

		wordcount_list = [word_id_list[word]['rawcount'] for word in word_id_list.keys()]
		wordcount = sum(wordcount_list)
		wordcount_max = max(wordcount_list)

		for word in word_id_list.keys():
			rawcount = word_id_list[word]['rawcount']
			doccount = word_id_list[word]['doccount']
			word_id_list[word]['raw_frequency'] = rawcount/wordcount
			word_id_list[word]['inverse_frequency'] = math.log(docsize/(1.0 + doccount))
			word_id_list[word]['tfidf'] = word_id_list[word]['raw_frequency'] * word_id_list[word]['inverse_frequency']

			print("raw: " + str(word_id_list[word]['raw_frequency']))
			print("inv: " + str(word_id_list[word]['inverse_frequency']))
			print("tfidf: " + str(word_id_list[word]['tfidf']))

		#input("a")
			

		with open(writedir_convert + "featurelist.txt", "r", encoding="utf-8") as f:
			data_read = [json.loads(data) for data in f.readlines()]
			data_train_x = []
			data_train_y = []

			for data in data_read:
				numseq = []
				dseq = data["sequence"]
				dclass = data["class"]
				for word in dseq:
					if word == SKIPWORD:
						numseq.append(0)
					else:
						if word in word_id_list.keys():
							numseq.append(word_id_list[word]['tfidf'])
						else:
							numseq.append(0)
				data_train_x.append(numseq)
				if (dclass == "comparative"):
					data_train_y.append(1)
				else:
					data_train_y.append(-1)
	

		print(data_train_x)
		print(data_train_y)

		clf = SVC(C = 1e5, kernel = 'linear')
		svmres = clf.fit(data_train_x, data_train_y)

		print('w = ',clf.coef_)
		print('b = ',clf.intercept_)
		print('Indices of support vectors = ', clf.support_)
		print('Support vectors = ', clf.support_vectors_)
		print('Number of support vectors for each class = ', clf.n_support_)
		#print('Coefficients of the support vector in the decision function = ', np.abs(clf.dual_coef_))

		with open(writedir_convert + "svm_results.txt", "wb") as f:
			f.write(pickle.dumps(svmres))

		with open(writedir_convert + "word_id_list.txt", "w", encoding="utf-8") as f:
			f.write(json.dumps(word_id_list))
	elif (args.mode == 'test'):
		with open(writedir_convert + "svm_results.txt", "rb") as f:
			svmdata = pickle.loads(f.read())
		with open(writedir_convert + "word_id_list.txt", 'r') as f:
			word_id_list = json.loads(f.read())


		#read the api files from this folder
		apifiles = [os.path.join(writedir_api_test,f) for f in os.listdir(writedir_api_test)]
		testsize = len(apifiles)

		keywordcorrect = {}
		for kw in KEYWORDS:
			keywordcorrect[kw] = {
				"compcorrect" : 0,
				"noncompcorrect" : 0,
				"compcount" : 0,
				"noncompcount" : 0,
				"total" : 0
			}
		keywordcorrect["-dan"] = {
				"compcorrect" : 0,
				"noncompcorrect" : 0,
				"compcount" : 0,
				"noncompcount" : 0,
				"total" : 0
		}

		compcorrect = 0
		noncompcorrect = 0
		compcount = 0
		noncompcount = 0

		featcount = 0


		for file in apifiles:
			with open(file) as f:
				features = []
				jsondata = [json.loads(line) for line in f.readlines()]
				#get class
				sentenceclass = jsondata.pop()

				#remove derivated words, they aren't part of the actual sentence
				jsondata_filter = [word for word in jsondata if (word["relation"] != "DERIV" and word["relation"] != "PUNCTUATION")]
				for idx,elem in enumerate(jsondata_filter):
					#check if the word is a keyword, or contains -dan,-den
					hasKeyword = (elem["word"] in KEYWORDS) or (elem["root"] in KEYWORDS)
					hasSuffix = re.search(KEYSUFFIX, elem["nounsuffixes"])

					if (hasKeyword or hasSuffix):

						keyval = ""

						min_elem = max(idx-SENTENCEOFFSET,0)
						max_elem = min(idx+SENTENCEOFFSET,len(jsondata_filter)-1)+1

						backappend = (SENTENCEOFFSET-idx) + min_elem
						frontappend = (idx + SENTENCEOFFSET) - max_elem + 1 

						#change the POS with a special POS
						if (hasKeyword):
							elem["pos1"] = elem["word"] + elem["pos1"]
							keyval = elem["root"]
						elif (hasSuffix):
							elem["pos1"] = KEYSUFFIX + elem["pos1"]
							keyval = "-dan"
						keywordcorrect[keyval]["total"] += 1

						sequence = [chosen_elem["pos1"] for chosen_elem in jsondata_filter[min_elem:max_elem]]

						#append default POS tags until the sequence sizes are equal
						for i in range(0,backappend):
							sequence.insert(0,"<S>")
						for i in range(0,frontappend):
							sequence.append("<S>")

						feature1 = {"sequence" : sequence, "class": sentenceclass["class"], "keyvalue" : keyval}

						features.append(feature1)
						featcount += 1

				for data in features:
					numseq = []
					dseq = data["sequence"]
					dclass = data["class"]
					keyval = data["keyvalue"]
					for word in dseq:
						if word == SKIPWORD:
							numseq.append(0)
						else:
							try:
								ind = word_id_list[word]
								numseq.append(ind['tfidf'])
							except KeyError:
								numseq.append(0)
					numseq2 = np.reshape(numseq, (1,-1)).astype(float)
					res = svmdata.predict(numseq2)
					res_str = ""

					if (dclass == "comparative"):
						compcount += 1
						keywordcorrect[keyval]["compcount"] += 1
					elif (dclass == "non-comparative"):
						noncompcount += 1
						keywordcorrect[keyval]["noncompcount"] += 1

					print(res[0])
					if (res[0] == 1):
						res_str = "comparative"
						if (dclass == res_str):
							compcorrect += 1
							keywordcorrect[keyval]["compcorrect"] += 1
					elif (res[0] == -1):
						res_str = "non-comparative"
						if (dclass == res_str):
							noncompcorrect += 1
							keywordcorrect[keyval]["noncompcorrect"] += 1

					print("Actual : " + dclass + ", Predicted: " + res_str)

		print("Total features: " + str(featcount))
		print("Correct Comparatives / Total: " + str(compcorrect) + " / " + str(compcount))
		print("Correct Non-comparatives / Total: " + str(noncompcorrect) + " / " + str(noncompcount))
		print("Accuracy: " + str( (compcorrect + noncompcorrect) / featcount))
		print("\n")

		for kw in keywordcorrect.keys():
			kwdict = keywordcorrect[kw]
			tot = kwdict["total"]
			com = kwdict["compcount"]
			ncom = kwdict["noncompcount"]
			comcor = kwdict["compcorrect"]
			ncomcor = kwdict["noncompcorrect"]
			print("Total sentences with " + kw + ": " + str(tot))
			print("Correct Comparatives / Total with " + kw + ": " + str(comcor) + "/" + str(com))
			print("Correct Non-comparatives / Total with " + kw + ": " + str(ncomcor) + "/" + str(ncom))
			if (tot == 0):
				print("Accuracy with " + kw + ": None")
			else:
				print("Accuracy with " + kw + ": " + str((comcor + ncomcor) / tot))
			print("\n")
			
		#print(svmdata.coef_)
		#print(svmdata.support_vectors_[0])

	else:
		print("Mode not specified.")