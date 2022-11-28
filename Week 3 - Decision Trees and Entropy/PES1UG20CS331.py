# Renita Kurian - PES1UG20CS331
# Lab 3 - Decision Tree Classifier

import numpy as np
import pandas as pd
import random

def get_entropy_of_dataset(df):
	last_column=list(df.columns)[-1]
	l=[]
	for i in df[last_column].unique():
		l.append(i)
	sum=0
	entropy=0
	for i in list(df[last_column]):
		if i in l:
			sum=sum+1
	for i in l:
		p_n=df[df.iloc[:,-1:]==i].count()[-1]
		if(p_n!=0):
			entropy = entropy-p_n/sum*np.log2(p_n/sum)
	return entropy
				

def get_avg_info_of_attribute(df, attribute):
	avg_info =0
	for n in set(df[attribute]):
		df_n = df[df[attribute]==n]
		p_n = df_n.shape[0]/df.shape[0]
		E = get_entropy_of_dataset(df_n)
		avg_info += (p_n*E)    	
	return avg_info

def get_information_gain(df, attribute):
	information_gain = get_entropy_of_dataset(df)-get_avg_info_of_attribute(df,attribute)
	return information_gain


def get_selected_attribute(df):
	d=dict()
	col=list(df.columns)
	col.pop()
	maxi=-1
	sel_col=''
	for i in col:
		d[i]=get_information_gain(df,i)
		if d[i]>maxi:
			maxi=d[i]
			sel_col=i
	return (d,sel_col)
	
	
	
	
	
	
