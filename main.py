from sklearn import tree 
from sklearn import metrics 
from sklearn import cross_validation 
import csv
import numpy 
import math

# Decision Tree classifier
# cross_validation is used for test and train data split
content=[]
file="F:/Projects/Gender Recognition by voice/voice.csv"
with open(file, 'r') as csvfile:
    read=csv.reader(csvfile)
    for row in read:
        content.append(row)


data=[]
digit=0
for i in content:
    if i[20]=='male':
        digit=0
    elif i[20]=='female':
        digit=1
    data.append([  i[0],i[2],i[3],i[4],i[8],i[10],i[11],i[13],i[14],i[16],i[19],i[5], i[17], i[7],i[18],i[12], i[6],i[15], i[9], digit ])

X=[]
y=[]
for i in data[1:]:
    X.append(i[:19])
    y.append(i[19])
    
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.33, random_state=0)
classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train,y_train)
expected=y_test
predict=classifier.predict(X_test)
score = metrics.accuracy_score(y_test, predict)
print score*100



