from __future__ import division, print_function
#from utils import NormalizationScaler, MinMaxScaler

from typing import List, Callable

import numpy as np
import scipy
from collections import Counter


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


class KNN:

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function
        #print("here")
        #print(k,distance_function)

    #TODO: Complete the training function
    def train(self, features: List[List[float]], labels: List[int]):
        #raise NotImplementedError
        self.train_features = features
        self.train_labels = labels
        #print("here")
        #print(features,labels)
        
    
    #TODO: Complete the prediction function
    def predict(self, features: List[List[float]]) -> List[int]:
        #raise NotImplementedError
        import scipy.stats
        features_train = self.train_features
        labels_train = self.train_labels
        k = self.k
        dist = np.full((np.array(features).shape[0],np.array(features_train).shape[0]), np.nan)
        for i in range(dist.shape[0]):
            for j in range(dist.shape[1]):
                dist[i,j] = self.distance_function(features[i],features_train[j])
        neighbors = dist.argsort()[:,:k]
        vote = np.full((np.array(features).shape[0],k), np.nan)
        for l in range(vote.shape[0]):
            for m in range(vote.shape[1]):
                vote[l,m] = labels_train[neighbors[l,m]]
        vote=vote.astype(int)
        predicted = scipy.stats.mode(vote,axis=1)[0][:,0].tolist()    
        return (predicted)
        
    #TODO: Complete the get k nearest neighbor function
    def get_k_neighbors(self, point):
        #raise NotImplementedError
        print("here")
        
            
if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
