import csv
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from data import getData
from sklearn.multiclass import OneVsOneClassifier
from matplotlib import pyplot as plt
import pandas as pd


def scaledata(data):
    l = list()
    for i in range(1, len(data[0])):
        ar = data[:, i]
        # print(ar)
        mean = ar.mean()
        # print(mean)
        std = ar.std()
        for j in range(len(data)):
            data[j][i] = (data[j][i]-mean)/std
    return data


# using data including orientation
file1 = open("data/cleanData.csv", 'r')
xx = csv.reader(file1, delimiter=" ")
dta = list()
for i in xx:
    dta.append(i)
dtaarr = np.array(dta)

# removing the orientation features
Data_NO_NT, Data_NO_NT_hand, Data_NO_NT_chest, Data_NO__NT_ankle = getData()
print(Data_NO_NT)


def runfrodiffdata(DataNONT):
    cc = [0.01, 0.1, 1, 10, 100, 1000]
    accuracypoly = []
    accuracyrbf = []
    accuracylinear = []
    print(Data_NO_NT)
    for i in range(6):
        X_train, X_test, y_train, y_test = train_test_split(
            DataNONT[:, 1:], DataNONT[:, 0], train_size=0.8, random_state=101)

        clfpoly = svm.SVC(
            kernel='poly', C=cc[i], decision_function_shape='ovo')
        # # modpoly = OneVsOneClassifier(clfpoly)
        clfrbf = svm.SVC(kernel='rbf', C=cc[i], decision_function_shape='ovo')
        clflin = svm.SVC(C=cc[i], kernel='linear',
                         decision_function_shape='ovo')

        clfpoly = clfpoly.fit(X_train, y_train)
        print("n")

        predpoly = clfpoly.predict(X_test)
        clfrbf = clfrbf.fit(X_train, y_train)
        print("n")

        predrbf = clfrbf.predict(X_test)
        clflin = clflin.fit(X_train, y_train)
        predlin = clflin.predict(X_test)

        accuracypoly.append(accuracy_score(y_true=y_test, y_pred=predpoly))
        accuracyrbf.append(accuracy_score(y_true=y_test, y_pred=predrbf))
        accuracylinear.append(accuracy_score(y_true=y_test, y_pred=predlin))
        print(i)
    df = pd.DataFrame({'C value': cc, 'rbf_accuracy': accuracyrbf,
                      'linear_accuracy': accuracylinear, 'poly_accuracy': accuracypoly})
    print(df)
    plt.plot(cc, accuracylinear)
    plt.plot(cc, accuracypoly)
    plt.plot(cc, accuracyrbf)
    # plt.legend()
    plt.show()


# runfrodiffdata(Data_NO_NT)
# runfrodiffdata(Data_NO__NT_ankle)
# runfrodiffdata(Data_NO_NT_chest)
# runfrodiffdata(Data_NO_NT_hand)

Data_NO_NT = scaledata(Data_NO_NT)
Data_NO_NT_hand = scaledata(Data_NO_NT_hand)
Data_NO_NT_chest = scaledata(Data_NO_NT_chest)
Data_NO__NT_ankle = scaledata(Data_NO__NT_ankle)

runfrodiffdata(Data_NO_NT)
# runfrodiffdata(Data_NO__NT_ankle)
# runfrodiffdata(Data_NO_NT_chest)
# runfrodiffdata(Data_NO_NT_hand)


print(len(dta))
