from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
def svm(kernel,c,X_train,y_train):
    svm_classifier = SVC(kernel=kernel,C=c)
    scores = cross_val_score(svm_classifier,X_train,y_train,cv=5)
    return scores.mean(),scores.std()