import os
import math
import json
import random
import sys
from shutil import copyfile


inputdir = os.getcwd() + "/apiresults"
traindir = inputdir + "/train"
testdir = inputdir + "/test"

def isComp(file):
	with open(inputdir + file, 'r') as f:
		jsondata = [json.loads(data) for data in f.readlines()]
		sentclass = jsondata.pop()['class']
		if (sentclass == "comparative"):
			return True
	return False


percentage = int(sys.argv[1])


for f in os.listdir(traindir):
    file_path = os.path.join(traindir, f)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)


for f in os.listdir(testdir):
    file_path = os.path.join(testdir, f)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)

files = ["/" + f for f in os.listdir(inputdir) if os.path.isfile(os.path.join(inputdir, f))]
compfiles = [f for f in files if isComp(f)]
noncompfiles = [f for f in files if not isComp(f)]
random.shuffle(noncompfiles)
random.shuffle(compfiles)


filecount = len(files)
compfilecount = len(compfiles)
noncompfilecount = len(noncompfiles)

count = 0;
while count < compfilecount * percentage / 100:
	file = compfiles[count]
	if (count < ( 38/40 * compfilecount * percentage / 100)):
		copyfile(inputdir + file, traindir + file)
	if (count > ( 30/40 * compfilecount * percentage / 100)):
		copyfile(inputdir + file, traindir + file)
		copyfile(inputdir + file, testdir + file)
	count += 1

count = 0;
while count < noncompfilecount * percentage / 100:
	file = noncompfiles[count]
	if (count < ( 11/40 * noncompfilecount * percentage / 100)):
		copyfile(inputdir + file, traindir + file)
	else:
		copyfile(inputdir + file, testdir + file)
	count += 1

