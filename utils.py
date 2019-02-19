import numpy as np
import matplotlib.pyplot as plt
from hw1_knn import KNN
from typing import List, Callable
np.seterr(divide='ignore',invalid='ignore')


'''def entropy(branches):
    branches = np.array(branches)
    totals = np.sum(branches, axis=0)
    #fractions = totals / np.sum(totals)
    entropy1 = branches / totals
    entropy1 = np.array([[-i * np.log2(i) if i > 0 else 0 for i in x] for x in entropy1])
    #print(entropy1,"sum")
    entropy1 = np.sum(entropy1, axis=0)
    print("entropy",entropy1)
    return entropy1'''
    
def Information_Gain(S,branches):
    # branches: List[List[any]]
    # return: float
    branches = np.array(branches)
    branches = branches.transpose()
    totals = np.sum(branches, axis=0)
    fractions = totals / np.sum(totals)
    entropy = branches / totals 
    entropy = np.array([[-i * np.log2(i) if i > 0 else 0 for i in x] for x in entropy])
    entropy = np.sum(entropy, axis=0)
    entropy = np.sum(entropy * fractions)
    return S - entropy

# TODO: implement reduced error pruning

#TODO: Information Gain function
#def Information_Gain(S, branches):
    # S: float
    # branches: List[List[any]]
    # return: float


# TODO: implement reduced error pruning
#def reduced_error_pruning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List[any]
    


# print current tree
# Do not change this function
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    for idx_cls in range(node.num_cls):
        string += str(node.labels.count(idx_cls)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')


#KNN Utils

#TODO: implement F1 score
def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    assert len(real_labels) == len(predicted_labels)

    tp = sum([x == 1 and y == 1 for x, y in zip(real_labels, predicted_labels)])
    fp = sum([x == 0 and y == 1 for x, y in zip(real_labels, predicted_labels)])
    fn = sum([x == 1 and y == 0 for x, y in zip(real_labels, predicted_labels)])
    if 2 * tp + fp + fn == 0:
        return 0
    f1 = 2 * tp / float(2 * tp + fp + fn)
    return f1
    
#TODO: Euclidean distance, inner product distance, gaussian kernel distance and cosine similarity distance

def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    distance = [(x - y) ** 2 for x, y in zip(point1, point2)]
    distance = np.sqrt(sum(distance))
    return distance

def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    distance = [(x * y) for x, y in zip(point1, point2)]
    distance = sum(distance)
    return distance

def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    distance = [(x - y) ** 2 for x, y in zip(point1, point2)]
    distance = sum(distance)
    distance = -np.exp(-0.5 * distance)
    return distance


def cosine_sim_distance(point1: List[float], point2: List[float]) -> float:
    d= np.dot(point1,point2)/(np.linalg.norm(point1)*np.linalg.norm(point2))
    return 1-d

def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval):
    best_f1_score, best_k = -1, 0
    best_function={}
    for name, func in distance_funcs.items():
        
        
        for k in range(1,30,2):
            if len(Xtrain)<k:
                break
                
            model = KNN(k=k, distance_function=func)
            model.train(Xtrain, ytrain)
            #print("model")
            #print(model.predict(Xtrain))
            train_f1_score = f1_score(ytrain, model.predict(Xtrain))
            valid_f1_score = f1_score(yval, model.predict(Xval))
            print('[part 1.1] {name}\tk: {k:d}\t'.format(name=name, k=k) + 
                  'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) +
                  'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))

            print()
            '''print('[part 2.1] {name}\tk: {k:d}\t'.format(name=name, k=k) + 
                  'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) +
                  'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))'''

            if valid_f1_score > best_f1_score:
                best_f1_score, best_k = valid_f1_score, k
                best_function=name

        model = KNN(k=best_k, distance_function=func)
        #print(Xtrain.shape,Xval.shape,ytrain.shape,yval.shape)
        model.train(np.concatenate((Xtrain,Xval),axis=0),np.concatenate((ytrain,yval),axis=0))
        '''test_f1_score = f1_score(ytest, model.predict(Xtest))
        print()
        print('[part 2.1] {name}\tbest_k: {best_k:d}\t'.format(name=name, best_k=best_k) +
          'test f1 score: {test_f1_score:.5f}'.format(test_f1_score=test_f1_score))
        print()
        print('[part 1.1] {name}\tbest_k: {best_k:d}\t'.format(name=name, best_k=best_k) +
              'test f1 score: {test_f1_score:.5f}'.format(test_f1_score=test_f1_score))
        print()'''
    return model,best_k,best_function


# TODO: select an instance of KNN with the best f1 score on validation dataset, with normalized data
def model_selection_with_transformation(distance_funcs, scaling_classes, Xtrain, ytrain, Xval, yval):
    #Dont change any print statement
    best_f1_score, best_k = 0, -1
    best_function={}
    best_scaler={}
    for scaling_name, scaling_class in scaling_classes.items():
        for name, func in distance_funcs.items():
            scaler = scaling_class()
            train_features_scaled = scaler(Xtrain)
            valid_features_scaled = scaler(Xval)

            
            for k in range(1,30,2):
                if len(Xtrain)<k:
                    break
                model = KNN(k=k, distance_function=func)
                model.train(train_features_scaled, ytrain)
                train_f1_score = f1_score(ytrain, model.predict(train_features_scaled))

                valid_f1_score = f1_score(yval, model.predict(Xval))
                print('[part 2.2] {name}\t{scaling_name}\tk: {k:d}\t'.format(name=name, scaling_name=scaling_name, k=k) +
                      'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) + 
                      'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))

                if valid_f1_score > best_f1_score:
                    best_f1_score, best_k = valid_f1_score, k
                    best_function=name
                    best_scaler=scaling_name


    # now change it to new scaler, since the training set changes
    scaler = scaling_classes.get(best_scaler)()
    combined_features_scaled = scaler(np.concatenate((Xtrain,Xval),axis=0))
    #test_features_scaled = scaler(X_test)

    model = KNN(k=best_k, distance_function=func)
    model.train(combined_features_scaled,np.concatenate((ytrain,yval),axis=0))
    '''test_f1_score = f1_score(ytest, model.predict(test_features_scaled))
            print()
            print('[part 2.2] {name}\t{scaling_name}\t'.format(name=name, scaling_name=scaling_name) +
                  'best_k: {best_k:d}\ttest: {test_f1_score:.5f}'.format(best_k=best_k, test_f1_score=test_f1_score))
            print()'''


    '''
    print('[part 1.2] {name}\t{scaling_name}\tk: {k:d}\t'.format(name=name, scaling_name=scaling_name, k=k) +
          'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) +
          'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))
    print()
    print('[part 1.2] {name}\t{scaling_name}\t'.format(name=name, scaling_name=scaling_name) +
          'best_k: {best_k:d}\ttest: {test_f1_score:.5f}'.format(best_k=best_k, test_f1_score=test_f1_score))
    print()'''
    return model,best_k,best_function,best_scaler
    
class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        normalized = []
        for sample in features:
            if all(x == 0 for x in sample):
                normalized.append(sample)
            else:
                denom = float(np.sqrt(inner_product_distance(sample, sample)))
                sample_normalized = [x / denom for x in sample]
                normalized.append(sample_normalized)
        return normalized

class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.
    Note:
        1. you may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).
    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]
        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]
        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]
        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]
        """

    def __init__(self):
        self.min = None
        self.max = None
    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        feat_array = np.array(features)
        
        if self.min is None or self.max is None:
            
            self.min = np.amin(feat_array, axis=0)
            self.max = np.amax(feat_array, axis=0)
            import collections
            '''if collections.Counter(self.min)==collections.Counter(self.max):
                normalized=[[0]*len(features)]*len(features[0])
                return normalized.tolist()
            else:'''
        normalized = (feat_array - self.min) / (self.max - self.min)
        return normalized.tolist()
        
