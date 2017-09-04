import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



train  = pd.read_csv("train.csv")  #lodd the training set
test = pd.read_csv("test.csv")     #load the test set
 
#check about data set
train.info()

#train.head()  #used to have a glimps of data set
drop = train.dropna().shape[0]  # calculates the number of non-missing items
l = train.shape[0] # calculates the number of items
nans = l - drop #calculates the missing values
'''print drop
print l
print nans'''


#Replacing NaN vlaues with some values
train.workclass.value_counts(sort=True)  #calculates the most occuring  we will get it as private so we replace every NaN with private  (use print(train.work.......) to get the value
train.workclass.fillna('Private',inplace=True)


#occupation
train.occupation.value_counts(sort=True)
train.occupation.fillna('Prof-specialty',inplace = True)


#Native Country

train['native.country'].value_counts(sort=True)
train['native.country'].fillna('United-States',inplace=True)

train.isnull().sum()  #used to calculate the number of NaN values


# convert the text to numbers
for x in train.columns:
    if train[x].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[x].values))
        train[x] = lbl.transform(list(train[x].values))
for x in test.columns:
    if test[x].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(test[x].values))
        test[x] = lbl.transform(list(test[x].values))
#print train['target']


#splitting data into inputs and outputs
X = train.iloc[:,:-1].values  
y = train.iloc[:,-1].values


#classifier
classifier = LogisticRegression(random_state = 0)
classifier.fit(X, y) # train


#splitting test data into inputs and outputs  
X_test = test.iloc[:,:-1].values
y_test = test.iloc[:,-1].values


#eval
y_pred = classifier.predict(X_test)


#make prediction and check model's accuracy
prediction = classifier.predict(X_test)
acc =  accuracy_score(y_test,prediction)
print ('The accuracy  {}'.format(acc))


