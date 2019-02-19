import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    # TODO: train Decision Tree
    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert(len(features) > 0)
        self.feautre_dim = len(features[0])
        num_cls = np.max(labels)+1
        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()
        return 

    # TODO: predic function
    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for feature in features:
            y_pred.append(self.root_node.predict(feature))
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
        # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    # TODO: implement split function
    def split(self):
        for idx_dim in range(len(self.features[0])):
        ############################################################
        # TODO: compare each split using conditional entropy
        #       find the 
        ############################################################

            if not 'max_entropy' in locals():
                max_entropy = -1
            xi = np.array(self.features)[:, idx_dim]
            if None in xi:
                continue
            branch_values = np.unique(xi)
            branches = np.zeros((len(branch_values), self.num_cls + 1))
            for i, val in enumerate(branch_values):
                y = np.array(self.labels)[np.where(xi == val)]
                for yi in y:
                    branches[i, yi] += 1
            e = 0
            X = np.unique(self.labels)
            for x in X:
                i = float(np.count_nonzero(self.labels == x)) / len(self.labels)
                e += i * np.log2(1/i)
                          
            info_gain_current = Util.Information_Gain(e, branches)
            if info_gain_current > max_entropy:
                #parent_entropy=Util.entropy(branches)
                max_entropy = info_gain_current
                self.dim_split = idx_dim
                self.feature_uniq_split = branch_values.tolist()

        ############################################################
        # TODO: split the node, add child nodes
        ############################################################
        xi = np.array(self.features)[:, self.dim_split]
        x = np.array(self.features, dtype=object)
        x[:, self.dim_split] = None
        # x = np.delete(self.features, self.dim_split, axis=1)
        for val in self.feature_uniq_split:
            indexes = np.where(xi == val)
            x_new = x[indexes].tolist()
            y_new = np.array(self.labels)[indexes].tolist()
            child = TreeNode(x_new, y_new, self.num_cls)
            if np.array(x_new).size == 0 or all(v is None for v in x_new[0]):
                child.splittable = False
            self.children.append(child)

        # split the child nodes
        for child in self.children:
            if child.splittable:
                child.split()

        return

    # TODO:treeNode predict function
    def predict(self, feature):
        if self.splittable:
            # print(feature)
            idx_child = self.feature_uniq_split.index(feature[self.dim_split])
            return self.children[idx_child].predict(feature)
        else:
            return self.cls_max