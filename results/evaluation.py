#!/usr/bin/env python
# Author: roman.klinger@ims.uni-stuttgart.de
# Evaluation script for Empathy shared task at WASSA 2021
# Adapted for CodaLab purposes by Orphee (orphee.declercq@ugent.be) in May 2018
# Adapted for multiple subtasks by Valentin Barriere in January 2020 (python 3)

from __future__ import print_function
import sys
import os
from math import sqrt

to_round = 4

def eprint(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)

def readFileToList(filename):
	#eprint("Reading data from",filename)
	lines=filename.readlines()
	result=[]
	for x in lines:
		result.append(x.rstrip().split('\t'))
	filename.close()
	return result

def calculatePRF(gold,prediction):
	"""
	gold/prediction list of list of emo predictions 
	"""
	# initialize counters
	labels = set(gold+prediction)
	tp = dict.fromkeys(labels, 0.0)
	fp = dict.fromkeys(labels, 0.0)
	fn = dict.fromkeys(labels, 0.0)
	precision = dict.fromkeys(labels, 0.0)
	recall = dict.fromkeys(labels, 0.0)
	f = dict.fromkeys(labels, 0.0)
	# check every element
	for g,p in zip(gold,prediction):
		# TP 
		if (g == p):
			tp[g] += 1
		else:
			fp[p] += 1
			fn[g] += 1
   # print("Label\tTP\tFP\tFN\tP\tR\tF")
	for label in labels:
		recall[label] = 0.0 if (tp[label]+fn[label]) == 0.0 else (tp[label])/(tp[label]+fn[label])
		precision[label] = 1.0 if (tp[label]+fp[label]) == 0.0 else (tp[label])/(tp[label]+fp[label])
		f[label] = 0.0 if (precision[label]+recall[label])==0 else (2*precision[label]*recall[label])/(precision[label]+recall[label])
		microrecall = (sum(tp.values()))/(sum(tp.values())+sum(fn.values()))
		microprecision = (sum(tp.values()))/(sum(tp.values())+sum(fp.values()))
		microf = 0.0 if (microprecision+microrecall)==0 else (2*microprecision*microrecall)/(microprecision+microrecall)
	# Macro average
	macrorecall = sum(recall.values())/len(recall)
	macroprecision = sum(precision.values())/len(precision)
	macroF = sum(f.values())/len(f)

	accuracy = 0
	for label in labels:
		accuracy += tp[label]

	accuracy = accuracy/len(gold)

	return round(microrecall,to_round),round(microprecision,to_round),round(microf,to_round),round(macrorecall,to_round),round(macroprecision,to_round),round(macroF,to_round),round(accuracy,to_round)

def pearsonr(x, y):
	"""
	Calculates a Pearson correlation coefficient. 
	"""

	assert len(x) == len(y), 'Prediction and gold standard does not have the same length...'

	xm = sum(x)/len(x)
	ym = sum(y)/len(y)

	xn = [k-xm for k in x]
	yn = [k-ym for k in y]

	r = 0 
	r_den_x = 0
	r_den_y = 0
	for xn_val, yn_val in zip(xn, yn):
		r += xn_val*yn_val
		r_den_x += xn_val*xn_val
		r_den_y += yn_val*yn_val

	r_den = sqrt(r_den_x*r_den_y)

	if r_den:
		r = r / r_den
	else:
		r = 0

	# Presumably, if abs(r) > 1, then it is only some small artifact of floating
	# point arithmetic.
	r = max(min(r, 1.0), -1.0)

	return round(r,to_round)

def calculate_pearson(gold, prediction):
	"""
	gold/prediction are a list of lists [ emp pred , distress pred ]
	"""

	# converting to float
	gold = [float(k) for k in gold]
	prediction = [float(k) for k in prediction]

	return pearsonr(gold, prediction)

def calculate_metrics(golds, predictions, task1, task2):
	"""
	gold/prediction list of list of values : [ emp pred , distress pred , emo pred ]
	"""
	if task1:
		gold_empathy = [k[1] for k in golds]
		prediction_empathy = [k[1] for k in predictions]
		pearson_empathy = calculate_pearson(gold_empathy, prediction_empathy)

		gold_distress = [k[1] for k in golds]
		prediction_distress = [k[1] for k in predictions]
		pearson_distress = calculate_pearson(gold_distress, prediction_distress)
		avg_pearson = (pearson_empathy + pearson_distress)/2
	else:
		avg_pearson, pearson_empathy, pearson_distress = 0,0,0

	if task2:
		gold_emo = [k[2] for k in golds]
		prediction_emo = [k[2] for k in predictions]

		microrecall,microprecision,microf,macrorecall,macroprecision,macroF,accuracy = calculatePRF(gold_emo, prediction_emo)
	else:
		microrecall,microprecision,microf,macrorecall,macroprecision,macroF,accuracy = 0,0,0,0,0,0,0

	return avg_pearson, pearson_empathy, pearson_distress, microrecall, microprecision, microf, macrorecall, macroprecision, macroF, accuracy

nb_labels_EMP = 2
nb_labels_EMO = 1

def score(input_dir, output_dir):
	# unzipped submission data is always in the 'res' subdirectory
	submission_path = os.path.join(input_dir, 'res', 'predictions_EMP.tsv')
	if not os.path.exists(submission_path):
		print('Could not find submission file {0}'.format(submission_path))
		task1 = False
	else:
		submission_file = open(os.path.join(submission_path))
		# The 2 first columns
		predictedList_EMP = [k[:nb_labels_EMP] for k in readFileToList(submission_file)]
		task1 = True

	# unzipped submission data is always in the 'res' subdirectory
	submission_path = os.path.join(input_dir, 'res', 'predictions_EMO.tsv')
	if not os.path.exists(submission_path):
		print('Could not find submission file {0}'.format(submission_path))
		predictedList = predictedList_EMP
		task2 = False
	else:
		submission_file = open(os.path.join(submission_path))
		predictedList_EMO = [k[:nb_labels_EMO] for k in readFileToList(submission_file)]

		if task1:
			# concatening the lists of preds
			predictedList = [i+j for i,j in zip(predictedList_EMP, predictedList_EMO)]
		else:
			predictedList = predictedList_EMO
		task2 = True

	# unzipped reference data is always in the 'ref' subdirectory
	truth_file = open(os.path.join(input_dir, 'ref', 'goldstandard.tsv'))
	goldList = readFileToList(truth_file)
	if (len(goldList) != len(predictedList)):
		eprint("Number of labels is not aligned!")
		sys.exit(1)

	avg_pearson, pearson_empathy, pearson_distress, micror, microp, microf, macror, macrop, macrof, accuracy = calculate_metrics(goldList,predictedList, task1, task2)

	with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
		str_to_write = ''
		# Not sure of that. Useful if the participant want to do only one subtask. Need to see if the leaderboard of the subtask does not update if there are nothing on score.txt 
		if task1:
			str_to_write += "Empathy Pearson Correlation: {0}\nDistress Pearson Correlation: {1}\nAveraged Pearson Correlations: {2}\n".format(pearson_empathy, pearson_distress, avg_pearson)
		if task2:
			str_to_write += "Micro Recall: {0}\nMicro Precision: {1}\nMicro F1-Score: {2}\nMacro Recall: {3}\nMacro Precision: {4}\nMacro F1-Score: {5}\nAccuracy: {6}\n".format(micror,microp,microf,macror,macrop,macrof,accuracy) 
		output_file.write(str_to_write)

def main():
	[_, input_dir, output_dir] = sys.argv
	score(input_dir, output_dir)

if __name__ == '__main__':
	main()