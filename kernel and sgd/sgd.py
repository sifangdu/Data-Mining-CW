from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier

import csv

X = []
y = []
with open('CreditCard_train.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    content = list(csv_reader)
    line_count = 0
    for row in content:
        if line_count > 1:
            X.append(row[1:24])
            y.append(int(row[24]))
            line_count+=1
        else:
            line_count+=1
    #convert string list to int     
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j] = int(X[i][j])


#X = [[0, 0, 0], [1, 1, 1], [1, 0, 1], [0, 1, 1]]
#y = [0, 0, 1, 1]

clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=50000)
clf.fit(X, y)   
clf.score(X, y)
#TEST AFTER HERE

Xtest = []
ytest = []


with open('CreditCard_test.csv', 'r') as csvfile2:
    csv_reader = csv.reader(csvfile2)
    content = list(csv_reader)
    line_count = 0
    for row in content:
        if line_count > 1:
            Xtest.append(row[1:24])
            ytest.append(int(row[24]))
            line_count+=1
        else:
            line_count+=1
    #convert string list to int
    for i in range(len(Xtest)):
        for j in range(len(Xtest[i])):
            Xtest[i][j] = int(Xtest[i][j])

iterations = 100
averagesum = 0

for i in range(iterations):
    correcthit = 0
    for x in range(len(Xtest)):
        predictedvalue = int(clf.predict([Xtest[x]]))
        if predictedvalue == ytest[x]:
            correcthit += 1
    averagesum+= correcthit/len(Xtest)


print("Over", iterations, "iterations the model achieved an average accuracy of", averagesum/iterations)




