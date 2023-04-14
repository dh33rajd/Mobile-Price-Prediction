#!/usr/bin/env python
# coding: utf-8

# # **importing data**

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

get_ipython().run_line_magic('matplotlib', 'inline')


# # **importing and splitting**

# In[2]:


dataset=pd.read_csv('Merged_Data.csv')
Train_Data=pd.read_csv('Train_Data.csv')
Traindata_classlabels=pd.read_csv('Traindata_classlabels.csv')
Train_Data_train, Train_Data_test, Traindata_classlabels_train, Traindata_classlabels_test = train_test_split(Train_Data, Traindata_classlabels, test_size=0.4, random_state=53)


# # **Data set visualization**



dataset.head()

dataset.isnull().sum()


dataset.info()
dataset.describe()


plt.figure(figsize=(20,20))
sns.heatmap(dataset.corr(),annot=True)
plt.show()


# # **training data**

# 
# 
# # **linear regression**



lm = LinearRegression()
lm.fit(Train_Data_train,Traindata_classlabels_train)
lm.score(Train_Data_test,Traindata_classlabels_test)
pred = lm.predict(Train_Data_test)
pred_acc = accuracy_score(Traindata_classlabels_test,pred)
pred_f = f1_score(Traindata_classlabels_test,pred,average='macro')
print("prediction accuracy = "+str(pred_acc))
print("prediction f measure = "+str(pred_f))
print(confusion_matrix(pred,Traindata_classlabels_test))


# # **K Nearest Neighbours**


arr=[]
for k in range(3,100,1):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Train_Data_train,Traindata_classlabels_train)
    pred = knn.predict(Train_Data_test)
    arr.append(f1_score(Traindata_classlabels_test,pred,average='macro'))



x=[]
for i in range(3,100,1):
    x.append(i)
plt.plot(x,arr)



plt.plot(x[10:20],arr[10:20])

knn = KNeighborsClassifier(n_neighbors=16)
knn.fit(Train_Data_train,Traindata_classlabels_train)
pred = knn.predict(Train_Data_test)
pred_acc = accuracy_score(Traindata_classlabels_test,pred)
pred_f = f1_score(Traindata_classlabels_test,pred,average='macro')
print("prediction accuracy = "+str(pred_acc))
print("prediction f measure = "+str(pred_f))
print(confusion_matrix(pred,Traindata_classlabels_test))


# # **Decision Tree**

clf = DecisionTreeClassifier(random_state=40) 
clf_parameters = {
            'criterion':('gini', 'entropy'), 
            'max_features':('auto', 'sqrt', 'log2',None),
            'max_depth':(15,30,45,60),
            'ccp_alpha':(0.009,0.005,0.05)
            } 
grid_search = GridSearchCV(estimator=clf,param_grid=clf_parameters,scoring='f1_macro',cv=5)
grid_search.fit(Train_Data_train,Traindata_classlabels_train)
print(grid_search.best_estimator_)
print("Decision tree score = ")
grid_search.best_estimator_.score(Train_Data_test,Traindata_classlabels_test)


pred = grid_search.best_estimator_.predict(Train_Data_test)
pred_acc = accuracy_score(Traindata_classlabels_test,pred)
pred_f = f1_score(Traindata_classlabels_test,pred,average='macro')
print(grid_search.best_estimator_)
print("prediction accuracy = "+str(pred_acc))
print("prediction f measure = "+str(pred_f))
print(confusion_matrix(pred,Traindata_classlabels_test))


a=0
arr=[]
x=[]
while(a<0.01):
    clf = DecisionTreeClassifier(ccp_alpha=a, criterion='entropy', max_depth=15,random_state=40)
    clf.fit(Train_Data_train,Traindata_classlabels_train)
    pred = clf.predict(Train_Data_test)
    arr.append(f1_score(Traindata_classlabels_test,pred,average='macro'))
    x.append(a)
    a=a+0.0001
    
plt.plot(x,arr)

plt.plot(x[40:60],arr[40:60])


clf = DecisionTreeClassifier(ccp_alpha=0.00425, criterion='entropy', max_depth=15,random_state=40)
clf.fit(Train_Data_train,Traindata_classlabels_train)
pred = clf.predict(Train_Data_test)
pred_acc = accuracy_score(Traindata_classlabels_test,pred)
pred_f = f1_score(Traindata_classlabels_test,pred,average='macro')
print("prediction accuracy = "+str(pred_acc))
print("prediction f measure = "+str(pred_f))
print(confusion_matrix(pred,Traindata_classlabels_test))


# # **Random forest classifier**


clf = RandomForestClassifier(n_estimators=200)
clf_parameters = {
            'criterion':('entropy','gini'),       
            'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 300, num = 100)],
            'max_depth':(10,20,30,50,100,200)
            } 
grid_search = GridSearchCV(estimator=clf,param_grid=clf_parameters,scoring='f1_macro',cv=5)
grid_search.fit(Train_Data_train,Traindata_classlabels_train)
print(grid_search.best_estimator_)
print("random forest score = ")
grid_search.best_estimator_.score(Train_Data_test,Traindata_classlabels_test)

pred = grid_search.best_estimator_.predict(Train_Data_test)
pred_acc = accuracy_score(Traindata_classlabels_test,pred)
pred_f = f1_score(Traindata_classlabels_test,pred,average='macro')
print(grid_search.best_estimator_)
print("prediction accuracy = "+str(pred_acc))
print("prediction f measure = "+str(pred_f))
print(confusion_matrix(pred,Traindata_classlabels_test))


# # **Gaussian Naive bayes**


clf = GaussianNB()
clf_parameters = {
            'var_smoothing':np.logspace(0,-13,num=100)
            }
grid_search = GridSearchCV(estimator=clf,param_grid=clf_parameters,scoring='f1_macro',cv=5)
grid_search.fit(Train_Data_train,Traindata_classlabels_train)
print(grid_search.best_estimator_)
print("gaussian score = ")
grid_search.best_estimator_.score(Train_Data_test,Traindata_classlabels_test)


from sklearn.metrics import precision_score
pred = grid_search.best_estimator_.predict(Train_Data_test)
pred_acc = accuracy_score(Traindata_classlabels_test,pred)
pred_f = f1_score(Traindata_classlabels_test,pred,average='macro')
print(grid_search.best_estimator_)
print("prediction accuracy = "+str(pred_acc))
print("prediction f measure = "+str(pred_f))
pred_prec = precision_score(Traindata_classlabels_test,pred,average='macro')
print("prediction precision = "+str(pred_prec))
print(confusion_matrix(pred,Traindata_classlabels_test))


# # **Support vector machine**

clf = svm.SVC(class_weight='balanced',probability=True)
clf_parameters = {
            'C':[0.01,0.1,1,10,100],
            'gamma': [1,0.1,0.01,0.001],
            'kernel':('linear','rbf','polynomial','sigmoid')
            }
grid_search = GridSearchCV(estimator=clf,param_grid=clf_parameters,scoring='f1_macro',cv=5)
grid_search.fit(Train_Data_train,Traindata_classlabels_train)
print(grid_search.best_estimator_)
print("svm score = ")
grid_search.best_estimator_.score(Train_Data_test,Traindata_classlabels_test)



pred = grid_search.best_estimator_.predict(Train_Data_test)
pred_acc = accuracy_score(Traindata_classlabels_test,pred)
pred_f = f1_score(Traindata_classlabels_test,pred,average='macro')
print(grid_search.best_estimator_)
print("prediction accuracy = "+str(pred_acc))
print("prediction f measure = "+str(pred_f))
print(confusion_matrix(pred,Traindata_classlabels_test))



a=0.0005
arr=[]
x=[]
while(a<0.02):
    clf = svm.SVC(C=a, class_weight='balanced', kernel='linear', probability=True)
    clf.fit(Train_Data_train,Traindata_classlabels_train)
    pred = clf.predict(Train_Data_test)
    arr.append(f1_score(Traindata_classlabels_test,pred,average='macro'))
    x.append(a)
    a=a+0.0005
    
plt.plot(x,arr)


plt.plot(x[0:100],arr[0:100])

np.max(arr)

print(arr[10],x[10])


clf = svm.SVC(C=0.0055, class_weight='balanced', kernel='linear', probability=True)
clf.fit(Train_Data_train,Traindata_classlabels_train)
pred = clf.predict(Train_Data_test)
pred_acc = accuracy_score(Traindata_classlabels_test,pred)
pred_f = f1_score(Traindata_classlabels_test,pred,average='macro')
precision = precision_score(Traindata_classlabels_test,pred,average='macro')
print("prediction precision = "+str(precision))
print("prediction accuracy = "+str(pred_acc))
print("prediction f measure = "+str(pred_f))
print(confusion_matrix(pred,Traindata_classlabels_test))


# # **Logistic regression**


clf = LogisticRegression(multi_class="multinomial",solver="lbfgs")
clf_parameters = {
     "C":np.logspace(-6,6,num=50,base=2),
     "penalty":["l1","l2",'elasticnet'],
     'solver':['newton-cg','lbfgs','liblinear']}
grid_search = GridSearchCV(estimator=clf,param_grid=clf_parameters,scoring='f1_macro',cv=5)
grid_search.fit(Train_Data_train,Traindata_classlabels_train)
print(grid_search.best_estimator_)
print("logistic regression score = ")
grid_search.best_estimator_.score(Train_Data_test,Traindata_classlabels_test)

# l1 lasso l2 ridge



print(grid_search.best_estimator_)



clf = LogisticRegression(C=0.015625, multi_class='multinomial', solver='newton-cg')
clf.fit(Train_Data_train,Traindata_classlabels_train)
pred = clf.predict(Train_Data_test)
pred_acc = accuracy_score(Traindata_classlabels_test,pred)
pred_f = f1_score(Traindata_classlabels_test,pred,average='macro')
precision = precision_score(Traindata_classlabels_test,pred,average='macro')
print("prediction precision = "+str(precision))
print("prediction accuracy = "+str(pred_acc))
print("prediction f measure = "+str(pred_f))
print(confusion_matrix(pred,Traindata_classlabels_test))

a=0.0005
arr=[]
x=[]
while(a<0.02):
    clf = LogisticRegression(C=a, multi_class='multinomial', solver='newton-cg')
    clf.fit(Train_Data_train,Traindata_classlabels_train)
    pred = clf.predict(Train_Data_test)
    arr.append(f1_score(Traindata_classlabels_test,pred,average='macro'))
    x.append(a)
    a=a+0.0005
    
plt.plot(x,arr)


np.max(arr)


print(arr[9],x[9])



clf = LogisticRegression(C=0.005, multi_class='multinomial', solver='newton-cg')
clf.fit(Train_Data_train,Traindata_classlabels_train)
pred = clf.predict(Train_Data_test)
pred_acc = accuracy_score(Traindata_classlabels_test,pred)
pred_f = f1_score(Traindata_classlabels_test,pred,average='macro')
precision = precision_score(Traindata_classlabels_test,pred,average='macro')
print("prediction precision = "+str(precision))
print("prediction accuracy = "+str(pred_acc))
print("prediction f measure = "+str(pred_f))
print(confusion_matrix(pred,Traindata_classlabels_test))



# # **Prediction of data**
# 
# We have found that the support vector machine has the highest f value as compared to other models used in this project. So I am using this model to predict the target values of the test data



Test_Data=pd.read_csv('testdata.csv')
clf=svm.SVC(C=0.0055, class_weight='balanced', kernel='linear', probability=True)
clf.fit(Train_Data_train,Traindata_classlabels_train)
predict = clf.predict(Test_Data)
predict






