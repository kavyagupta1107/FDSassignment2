import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import math
from sklearn.metrics import classification_report


def DTWDistance(s1, s2,w):
    DTW={}
    
    w = max(w, abs(len(s1)-len(s2)))
    
    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0
  
    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
		
    return math.sqrt(DTW[len(s1)-1, len(s2)-1])


def euclid_dist(t1,t2):
    return math.sqrt(sum((t1-t2)**2))

def funcc(s1,s2,r):
    LB_sum=0
    for ind,i in enumerate(s1):
        
        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        
        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2
    
    return math.sqrt(LB_sum)

ls =[]
def knn(train,test,w):
    preds=[]
    for ind,i in enumerate(test):
        min_dist=float('inf')
        closest_seq=[]
        #print ind
        for j in train:
            if funcc(i[:-1],j[:-1],5)<min_dist:
                dist=DTWDistance(i[:-1],j[:-1],w)
                if dist<min_dist:
                    min_dist=dist
                    closest_seq=j
                    print(closest_seq[-1])
                    ls.append(closest_seq[-1])
                    preds.append(closest_seq[-1])
    return classification_report(test[:,-1],preds)

def my_accuracy(y_pred, vl):
  cnt =0
  for j in range(len(y_pred)):
    if(y_pred[j] == vl[j]):
      cnt = cnt + 1
  return (cnt/len(y_pred)* 100)

train = np.genfromtxt('fdstrain.csv', delimiter=',')
test = np.genfromtxt('fdstest.csv', delimiter=',')

print(knn(train,test[:,0:5],4))
