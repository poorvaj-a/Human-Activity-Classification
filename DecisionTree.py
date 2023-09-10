from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocess import Data_NO_NT,Data_NO_NT_chest,Data_NO__NT_ankle,Data_NO_NT_hand
import pickle
import numpy as np
from scipy import stats
import random
import math
from sklearn.model_selection import KFold
import time
from sklearn.ensemble import AdaBoostClassifier
import statistics as st

#add cross validations too

def SingleTree(x_train,y_train,x_test,y_test,max_depth):
    Tree = tree.DecisionTreeClassifier(max_depth=max_depth)
    Tree.fit(x_train,y_train)
    predictions = Tree.predict(x_test)
    file = open("Dt.pkl",'wb')
    pickle.dump(Tree,file)
    return accuracy_score(y_test,predictions)
def scaledata(data):
    for i in range(1, len(data[0])):
        ar = data[:, i]
        # print(ar)
        mean = ar.mean()
        std = ar.std()
        for j in range(len(data)):
            data[j][i] = (data[j][i]-mean)/std
    return data
Data_NO_NT = scaledata(Data_NO_NT)
X=Data_NO_NT[:,1:]
y=Data_NO_NT[:,0]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=9)
# Data_NO_NT_hand = scaledata(Data_NO_NT_hand)
# X=Data_NO_NT_hand[:,1:]
# y=Data_NO_NT_hand[:,0]
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=9)
### may need to use stratification since number of zeroes is very large
# print(SingleTree(X_train,y_train,X_test,y_test,0))

class Bagging:
    def __init__(self,X,y,num_trees,max_depth):
        self.X=X
        self.y=y
        self.num_trees=num_trees
        self.max_depth=max_depth
        self.DecisionTrees = []
        
        
    def random_sampler(self,N):
        return random.sample(range(0,N),N)
    
    
    def fit(self):
        for i in range(0,self.num_trees):
            newTree = tree.DecisionTreeClassifier(splitter='random',max_depth=self.max_depth,max_features="sqrt")
            sampled_idx = self.random_sampler(len(self.X))
            newTree.fit(self.X[sampled_idx],self.y[sampled_idx])
            self.DecisionTrees.append(newTree)

    def predict(self,x_test):
        y_pred = []
        for x in x_test:
            y_trees = np.zeros(self.num_trees)
            for i in range(0,self.num_trees):
                y_trees[i] = self.DecisionTrees[i].predict([x])[0]
            y_pred.append(st.mode(y_trees))
        return y_pred

##################################################
#Do not forget to use 1 standard error rule in cv
#Do not forget to use 1 standard error rule in cv
#Do not forget to use 1 standard error rule in cv
#Do not forget to use 1 standard error rule in cv
#Do not forget to use 1 standard error rule in cv
#Do not forget to use 1 standard error rule in cv
#Do not forget to use 1 standard error rule in cv
#Do not forget to use 1 standard error rule in cv
#Do not forget to use 1 standard error rule in cv
#####################################################

class Boosting:
    def __init__(self,X,y,num_trees,max_depth,num_classes):
        self.X=X
        self.y=y
        self.num_trees=num_trees
        self.max_depth=max_depth
        self.DecisionTrees = []
        self.tree_weights = []
        self.weights = np.ones(X.shape[0])*1/X.shape[0]
        self.num_classes = num_classes
    
    def fit(self):
        for i in range(0,self.num_trees):
            print(i)
            print(time.time())
            newTree = tree.DecisionTreeClassifier(max_depth=self.max_depth)
            
            newTree.fit(self.X,self.y,sample_weight=self.weights)
            print(time.time())
            self.DecisionTrees.append(newTree)
            y_pred = newTree.predict(self.X)
            error = 0
            for j in range(0,len(y_pred)):
                error+= self.weights[j]*int((y_pred[j] != self.y[j]))
            error = error/np.sum(self.weights)
            self.tree_weights.append(math.log((1-error)/error))
            for j in range(0,len(self.weights)):
                self.weights[j] = self.weights[j] * math.exp(self.tree_weights[i]*int(y_pred[j] != y[j]))
                
            
    def predict(self,x_test):
        y_pred = []
        for x in x_test:
            class_weight = np.zeros(self.num_classes)
            for i in range(0,self.num_trees):
                class_weight[int(self.DecisionTrees[i].predict([x])[0])] += self.tree_weights[i] 
            y_pred.append(int(np.argmax(class_weight)))
        return y_pred

# k=5
# kfold=KFold(n_splits=5,shuffle=True,random_state=3)
# for i in [50,100,250,500]:
#     for j in range(1,11):
#         for train, test in kfold.split(X_train):
#             boosted = Boosting(X_train[train],y_train[train],i,j,25)
#             boosted.fit()
#             print("Trees",i,"depth",j,accuracy_score(y_train[test],boosted.predict(X_train[test])))


# accuracies = np.zeros((4))
# index = 0
# for i in [100]:
#     for j in range(5,21,5):
#         print(time.time())
#         bagging = Bagging(X_train,y_train,i,j)
#         bagging.fit()
#         print(time.time())
#         accuracies[index]= accuracy_score(y_test,bagging.predict(X_test))
#         print("Trees",i,"depth",j,accuracies[index])
#         index+=1
for depth in range(5,40,5):
    print("depth",depth,SingleTree(X_train,y_train,X_test,y_test,depth))
    
    
            
