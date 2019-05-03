import os
import json

curdir = os.getcwd() + "/train"


files = [os.path.join(curdir,f) for f in os.listdir(curdir)]

com = 0
noncom = 0

for file in files:
	with open(file, 'r') as f:
		jsondata = [json.loads(data) for data in f.readlines()]
		sentclass = jsondata.pop()['class']
		if (sentclass == "comparative"):
			com += 1
		elif (sentclass == "non-comparative"):
			noncom += 1

print(com)
print(noncom)