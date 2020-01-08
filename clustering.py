from tslearn.generators import random_walks
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler
from tslearn.utils import to_time_series_dataset
import pandas as pd
import numpy as np

# X_bis = to_time_series_dataset([[[1, 2, 3, 4],[1, 2, 3, 4],[2, 5, 6, 7]],[[1, 7, 9, 4],[2, 2, 3, 4],[6, 12, 6, 2]],[[1, 1, 3, 2],[1, 3, 3, 2],[2, 8, 6, 7]]])
df = pd.read_csv('cluster.csv')
print(df.head())
X = []
for i in range(0,100):
  pd1 = df.iloc[4*i:4*i+4,2:]
  print(pd1.head())
  X.append(pd1.values)
print(X[0])

X_k = to_time_series_dataset(X)
k = TimeSeriesKMeans(n_clusters=5, max_iter=10,metric="dtw", random_state=0).fit(X_k)

print(k.cluster_centers_.shape)
print(k.cluster_centers_)