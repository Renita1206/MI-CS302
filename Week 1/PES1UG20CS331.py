#This weeks code focuses on understanding basic functions of pandas and numpy 
#This will help you complete other lab experiments
import random

# Do not change the function definations or the parameters
from calendar import c
import numpy as np
import pandas as pd

#input: tuple (x,y)    x,y:int 
def create_numpy_ones_array(shape):
	#return a numpy array with one at all index
	array=np.ones(shape)
	return array
#print(create_numpy_ones_array((2,4)))

#input: tuple (x,y)    x,y:int 
def create_numpy_zeros_array(shape):
	#return a numpy array with zeros at all index
	array=np.zeros(shape)
	#TODO
	return array
#print(create_numpy_zeros_array((2,4)))

#input: int  
def create_identity_numpy_array(order):
	#return a identity numpy array of the defined order
    array=np.zeros((order,order))
    for i in range(order):
        array[i][i] = 1
    return array
#print(create_identity_numpy_array(3))

def fill_with_mode(filename, column):
    df=pd.read_csv(filename)
    df[column] = df[column].fillna(df[column].mode()[0])
    return df

def fill_with_group_average(df, group, column):
    df[column] = df[column].fillna(df.groupby(group)[column].transform('mean'))
    #print(df[column])
    return df


def get_rows_greater_than_avg(df, column):
    df = df[df[column] > df[column].mean()]
    #print(df.head())
    return df

#Input: (numpy array, int ,numpy array, int , int , int , int , tuple,tuple)
#tuple (x,y)    x,y:int 
def f1(X1,coef1,X2,coef2,seed1,seed2,seed3,shape1,shape2):
	#note: shape is of the forst (x1,x2)
	#return W1 x (X1 ** coef1) + W2 x (X2 ** coef2) +b
	# where W1 is random matrix of shape shape1 with seed1
	# where W2 is random matrix of shape shape2 with seed2
	# where B is a random matrix of comaptible shape with seed3
	# if dimension mismatch occur return -1
    (a,b) = shape1
    (c,d) = shape2
    if(a!=c):
        return -1
    random.seed(seed1)
    w1 = np.random.rand(a,b)
    random.seed(seed2)
    w2 = np.random.rand(c,d)
    a1 = np.power(X1, coef1)
    a2 = np.power(X2, coef2)
    random.seed(seed3)
    b = np.random.rand(a, d)
    ans = np.matmul(w1, a1) + np.matmul(w2, a2) + b
    return ans

#-------------------------------------------------------------------
#input: numpy array
def matrix_cofactor(array):
    arr = array
    for i in range(0,2):
        for j in range(0,2):
            arr[i][j] = ((-1)**(i+j)) * array[j][i]
    #print(arr)
    return arr

