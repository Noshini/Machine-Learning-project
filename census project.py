#!/usr/bin/env python
# coding: utf-8

# Problem Statement :
#     To build a model that will predict if the income of any individual in the US is greater than or less than 
#     USD 50,000 based on the data available about the individual.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data=pd.read_csv(r"C:\Users\HP\Downloads\census - census.csv")
data.head()


# In[3]:


data.shape


# In[4]:


data.isnull().sum()


# In[5]:


data.dtypes


# In[6]:


data.columns


# In[7]:


data.info()


# In[8]:


data.income.value_counts()


# In[9]:


from sklearn.preprocessing import LabelEncoder
la=LabelEncoder()


# In[10]:


l=["age","workclass","education","occupation","race","sex","capital.gain",
   "capital.loss","hours.per.week","native.country","income","marital.status","relationship"]

for i in l:
    data[i]=la.fit_transform(data[i])


# In[11]:


data=data.rename(columns={"capital.gain":"capital_gain","capital.loss":"capital_loss","hours.per.week":"hours_per_week",
                          "native.country":"native_country","marital.status":"marital_status","education.num":"education_num"})
data.head(2)


# In[12]:


data.dtypes


# In[13]:


data.head(2)


# In[14]:


#Classification


# In[15]:


x=data.drop(columns=['income'])
y=data['income'] #target column


# In[16]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)


# In[17]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# # BASE MODEL

# # Logistic Regression

# In[18]:


from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()


# In[19]:


lg.fit(x_train,y_train)


# In[20]:


p1=lg.predict(x_test)
p1


# In[21]:


from sklearn.metrics import confusion_matrix
lg_c=confusion_matrix(y_test,p1) #p1 predicted value for y_test
lg_c


# In[22]:


lg_ac=lg_c.diagonal().sum()/lg_c.sum()*100
lg_ac


# In[23]:


from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score


# In[24]:


print(classification_report(y_test,p1))


# In[25]:


lg_precision=precision_score(y_test,p1)
lg_recall=recall_score(y_test,p1)
lg_f1_score=f1_score(y_test,p1)
print(lg_precision,lg_recall,lg_f1_score)


# # Decision tree

# In[26]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()


# In[27]:


dt.fit(x_train,y_train)


# In[28]:


dt_p=dt.predict(x_test)
dt_p


# In[29]:


dt_c=confusion_matrix(y_test,dt_p)
dt_c


# In[30]:


dt_ac=dt_c.diagonal().sum()/dt_c.sum()*100
dt_ac


# In[31]:


print(classification_report(y_test,dt_p))


# In[32]:


dt_precision=precision_score(y_test,dt_p)
dt_recall=recall_score(y_test,dt_p)
dt_f1_score=f1_score(y_test,dt_p)
print(dt_precision,dt_recall,dt_f1_score)


# # svm

# In[33]:


from sklearn.svm import SVC
svm=SVC()


# In[34]:


svm.fit(x_train,y_train)


# In[35]:


svm_p=svm.predict(x_test)


# In[36]:


svm_cf=confusion_matrix(y_test,svm_p)
svm_cf


# In[37]:


svm_ac=svm_cf.diagonal().sum()/svm_cf.sum()*100
svm_ac


# In[38]:


print(classification_report(y_test,svm_p))


# In[39]:


svm_precision=precision_score(y_test,svm_p)
svm_recall=recall_score(y_test,svm_p)
svm_f1_score=f1_score(y_test,svm_p)
print(svm_precision,svm_recall,svm_f1_score)


# # KNN

# In[40]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()


# In[41]:


knn.fit(x_train,y_train)


# In[42]:


knn_p=knn.predict(x_test)


# In[43]:


knn_c=confusion_matrix(y_test,knn_p)
knn_c


# In[44]:


knn_ac=knn_c.diagonal().sum()/knn_c.sum()*100
knn_ac


# In[45]:


print(classification_report(y_test,knn_p))


# In[46]:


knn_precision=precision_score(y_test,knn_p)
knn_recall=recall_score(y_test,knn_p)
knn_f1_score=f1_score(y_test,knn_p)
print(knn_precision,knn_recall,knn_f1_score)


# # XGB

# In[47]:


import xgboost


# In[48]:


from xgboost import XGBClassifier
xgb=XGBClassifier()


# In[49]:


xgb.fit(x_train,y_train)


# In[50]:


xgb_p=xgb.predict(x_test)


# In[51]:


df=pd.DataFrame(xgb_p)


# In[52]:


xgb_cf=confusion_matrix(y_test,xgb_p)
xgb_cf


# In[53]:


xgb_ac=xgb_cf.diagonal().sum()/xgb_cf.sum()*100
xgb_ac


# In[54]:


print(classification_report(y_test,xgb_p))


# In[55]:


xgb_precision=precision_score(y_test,xgb_p)
xgb_recall=recall_score(y_test,xgb_p)
xgb_f1_score=f1_score(y_test,xgb_p)
print(xgb_precision,xgb_recall,xgb_f1_score)


# # Naive Bayes

# In[56]:


from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()


# In[57]:


nb.fit(x_train,y_train)


# In[58]:


nb_p=nb.predict(x_test)


# In[59]:


nb_cf=confusion_matrix(y_test,nb_p)
nb_cf


# In[60]:


nb_ac=nb_cf.diagonal().sum()/nb_cf.sum()*100
nb_ac


# In[61]:


print(classification_report(y_test,nb_p))


# In[62]:


nb_precision=precision_score(y_test,nb_p)
nb_recall=recall_score(y_test,nb_p)
nb_f1_score=f1_score(y_test,nb_p)
print(nb_precision,nb_recall,nb_f1_score)


# # RF

# In[63]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()


# In[64]:


rf.fit(x_train,y_train)


# In[65]:


rf_p=rf.predict(x_test)


# In[66]:


rf_c=confusion_matrix(y_test,rf_p)
rf_c


# In[67]:


rf_ac=rf_c.diagonal().sum()/rf_c.sum()*100
rf_ac


# In[68]:


print(classification_report(y_test,rf_p))


# In[69]:


rf_precision=precision_score(y_test,rf_p)
rf_recall=recall_score(y_test,rf_p)
rf_f1_score=f1_score(y_test,rf_p)
print(rf_precision,rf_recall,rf_f1_score)


# In[70]:


L1=["Logistic","DT","SVM","KNN","XGB","NB","RF"]
L2=[lg_ac,dt_ac,svm_ac,knn_ac,xgb_ac,nb_ac,rf_ac]
L3=[lg_precision,dt_precision,svm_precision,knn_precision,xgb_precision,nb_precision,rf_precision]
L4=[lg_recall,dt_recall,svm_recall,knn_recall,xgb_recall,nb_recall,rf_precision]
L5=[lg_f1_score,dt_f1_score,svm_f1_score,knn_f1_score,xgb_f1_score,nb_f1_score,rf_f1_score]


# In[71]:


final_df=pd.DataFrame({"Model":L1,"Accuracy":L2,"Precision":L3,"Recall":L4,"F1_score":L5})
final_df


# In[72]:


#Conclusion :  RF gives accuracy, precision, recall and f1-score is good as compared to other models.


# In[73]:


#Confusion matrix for XGB model because the performance of the model is good as compared to other models
cm=confusion_matrix(y_test,xgb_p)
sns.heatmap(cm,annot=True,fmt="d",cmap="Spectral")
plt.title("Confusion Matrix")
plt.show()


# In[74]:


#AUC-ROC CURVE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# In[75]:


roc=roc_auc_score(xgb_p,y_test)


# In[76]:


fpr,tpr,threshold=roc_curve(y_test,df.iloc[:,-1])


# In[77]:


plt.plot(fpr,tpr)


# In[78]:


#PRC Curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score


# In[79]:


precision,recall,threshold=precision_recall_curve(y_test,df.iloc[:,-1])


# In[80]:


plt.plot(recall,precision,label="Precision-Recall")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="upper left")
plt.title("PRC-CURCE")


# # Class Imbalance - Random Oversampling

# In[81]:


sns.countplot(data=data,y="income")


# In[82]:


c_0,c_1=data["income"].value_counts()
c_0,c_1


# In[83]:


train_0=data[data["income"]==0]
train_1=data[data["income"]==1]
train_0.shape,train_1.shape


# In[84]:


#train_0.head(2)
#train_1.head(2)


# In[85]:


class_1over=train_1.sample(c_0,replace=True)


# In[86]:


class_1over.head()
class_1over.shape,train_0.shape


# In[87]:


class1_0=pd.concat([class_1over,train_0],axis=0)
class1_0.shape


# In[88]:


class1_0.income.value_counts()


# In[89]:


class1_0.skew()


# In[90]:


import matplotlib.pyplot as plt
import seaborn as sns

num_cols = ['age', 'fnlwgt', 'hours_per_week']
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))

for i, col in enumerate(num_cols):
    sns.boxplot(class1_0[col], ax=axes[i])  # Plot boxplot for the current column
    axes[i].set_title(f'Boxplot for {col}')  # Set title for the current subplot

plt.tight_layout()
plt.show()


# # Outliers

# In[91]:


sns.distplot(class1_0.age)#sd outlier & iqr


# In[92]:


sns.boxplot(data=class1_0,y="age")


# In[93]:


ub=class1_0.age.mean()+3*class1_0.age.std()
lb=class1_0.age.mean()-3*class1_0.age.std()
print(ub,lb)


# In[94]:


class1_0.loc[class1_0["age"]>62.32158418958262,"age"]=62.32158418958262


# In[95]:


sns.boxplot(data=class1_0,y="age")


# In[96]:


sns.distplot(class1_0.age)


# In[97]:


sns.distplot(class1_0.hours_per_week)


# In[98]:


sns.boxplot(data=class1_0,y="hours_per_week") #sd


# In[99]:


ub1=class1_0.hours_per_week.mean()+3*class1_0.hours_per_week.std()
lb1=class1_0.hours_per_week.mean()-3*class1_0.hours_per_week.std()
print(ub1,lb1)


# In[100]:


class1_0.loc[class1_0["hours_per_week"]>76.83096060089193,"hours_per_week"]=76.83096060089193


# In[101]:


sns.boxplot(data=class1_0,y="hours_per_week")


# In[102]:


sns.distplot(class1_0.hours_per_week)


# In[103]:


#skewness treatment


# In[104]:


class1_0.skew()


# In[105]:


sns.distplot(class1_0.fnlwgt)


# In[106]:


class1_0.fnlwgt.hist()


# In[107]:


class1_0.fnlwgt.describe()


# In[108]:


IQR1=class1_0.fnlwgt.quantile(0.75)-class1_0.fnlwgt.quantile(0.25)
IQR1


# In[109]:


ub2=class1_0.fnlwgt.quantile(0.75)+1.5*IQR1
lb2=class1_0.fnlwgt.quantile(0.25)-1.5*IQR1
print(ub2,lb2)


# In[110]:


class1_0.loc[class1_0["fnlwgt"]>234928,"fnlwgt"]=234928


# In[111]:


sns.distplot(class1_0.fnlwgt)


# In[112]:


class1_0.fnlwgt.hist()


# In[113]:


class1_0.skew()


# In[114]:


x1=class1_0.iloc[:,1:-1]
y1=class1_0.iloc[:,-1]


# In[115]:


x1_tr,x1_te,y1_tr,y1_te=train_test_split(x1,y1,test_size=0.2,random_state=100)
x1_tr.shape,x1_te.shape,y1_tr.shape,y1_te.shape


# # Logistic Regression - Oversampling

# In[116]:


lg.fit(x1_tr,y1_tr)


# In[117]:


op1=lg.predict(x1_te)


# In[118]:


oc1=confusion_matrix(y1_te,op1)
oc1


# In[119]:


oa1=oc1.diagonal().sum()/oc1.sum()*100
oa1


# In[120]:


print(classification_report(y1_te,op1))


# In[121]:


lg2_precision=precision_score(y1_te,op1)
lg2_recall=recall_score(y1_te,op1)
lg2_f1_score=f1_score(y1_te,op1)
print(lg2_precision,lg2_recall,lg2_f1_score)


# # Decision Tree - Oversampling

# In[122]:


dt.fit(x1_tr,y1_tr)


# In[123]:


op2=dt.predict(x1_te)


# In[124]:


oc2=confusion_matrix(y1_te,op2)
oc2


# In[125]:


oa2=oc2.diagonal().sum()/oc2.sum()*100
oa2


# In[126]:


print(classification_report(y1_te,op2))


# In[127]:


dt2_precision=precision_score(y1_te,op2)
dt2_recall=recall_score(y1_te,op2)
dt2_f1_score=f1_score(y1_te,op2)
print(dt2_precision,dt2_recall,dt2_f1_score)


# # Random Forest - Oversampling

# In[128]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()


# In[129]:


rf.fit(x1_tr,y1_tr)


# In[130]:


op3=rf.predict(x1_te)


# In[131]:


df1=pd.DataFrame(op3)


# In[132]:


oc3=confusion_matrix(y1_te,op3)
oc3


# In[133]:


oa3=oc3.diagonal().sum()/oc3.sum()*100
oa3


# In[134]:


print(classification_report(y1_te,op3))


# In[135]:


rf2_precision=precision_score(y1_te,op3)
rf2_recall=recall_score(y1_te,op3)
rf2_f1_score=f1_score(y1_te,op3)
print(rf2_precision,rf2_recall,rf2_f1_score)


# # KNN - Oversampling

# In[136]:


knn.fit(x1_tr,y1_tr)


# In[137]:


op4=knn.predict(x1_te)


# In[138]:


oc4=confusion_matrix(y1_te,op4)
oc4


# In[139]:


oa4=oc4.diagonal().sum()/oc4.sum()*100
oa4


# In[140]:


print(classification_report(y1_te,op4))


# In[141]:


knn2_precision=precision_score(y1_te,op4)
knn2_recall=recall_score(y1_te,op4)
knn2_f1_score=f1_score(y1_te,op4)
print(knn2_precision,knn2_recall,knn2_f1_score)


# # XGB

# In[142]:


xgb.fit(x1_tr,y1_tr)


# In[143]:


op5=xgb.predict(x1_te)


# In[144]:


oc5=confusion_matrix(y1_te,op5)
oc5


# In[145]:


oa5=oc5.diagonal().sum()/oc5.sum()*100
oa5


# In[146]:


print(classification_report(y1_te,op5))


# In[147]:


xgb2_precision=precision_score(y1_te,op5)
xgb2_recall=recall_score(y1_te,op5)
xgb2_f1_score=f1_score(y1_te,op5)
print(xgb2_precision,xgb2_recall,xgb2_f1_score)


# # SVM

# In[148]:


svm.fit(x1_tr,y1_tr)


# In[149]:


op6=svm.predict(x1_te)


# In[150]:


oc6=confusion_matrix(y1_te,op6)
oc6


# In[151]:


oa6=oc6.diagonal().sum()/oc6.sum()*100
oa6


# In[152]:


print(classification_report(y1_te,op6))


# In[153]:


svm2_precision=precision_score(y1_te,op6)
svm2_recall=recall_score(y1_te,op6)
svm2_f1_score=f1_score(y1_te,op6)
print(svm2_precision,svm2_recall,svm2_f1_score)


# # Naive Bayes

# In[154]:


nb.fit(x1_tr,y1_tr)


# In[155]:


op7=nb.predict(x1_te)


# In[156]:


oc7=confusion_matrix(y1_te,op7)
oc7


# In[157]:


oa7=oc7.diagonal().sum()/oc7.sum()*100
oa7


# In[158]:


print(classification_report(y1_te,op7))


# In[159]:


nb2_precision=precision_score(y1_te,op7)
nb2_recall=recall_score(y1_te,op7)
nb2_f1_score=f1_score(y1_te,op7)
print(nb2_precision,nb2_recall,nb2_f1_score)


# In[160]:


LO11=["Logistic","DT","RF","KNN","XGB","SVM","NB"]
LO22=[oa1,oa2,oa3,oa4,oa5,oa6,oa7]
LO33=[lg2_precision,dt2_precision,rf2_precision,knn2_precision,xgb2_precision,svm2_precision,nb2_precision]
LO44=[lg2_recall,dt2_recall,rf2_recall,knn2_recall,xgb2_recall,svm2_recall,nb2_recall]
LO55=[lg2_f1_score,dt2_f1_score,rf2_f1_score,knn2_f1_score,xgb2_f1_score,svm2_f1_score,nb2_f1_score,]


# In[161]:


final_df1=pd.DataFrame({"Model":LO11,"Accuracy":LO22,"Precision":LO33,"Recall":LO44,"F1_score":LO55})
final_df1


# In[162]:


#Conclusion : The accuracy of the model increases as compared to Base Model. 
#The accuracy and AUC are close to each other, in this case RF model is fitted. 


# In[163]:


#Confusion matrix for RF model because the performance of the model is good as compared to other models
cm1=confusion_matrix(y1_te,op3)
sns.heatmap(cm1,annot=True,fmt="d",cmap="Spectral")
plt.title("Confusion Matrix")
plt.show()


# In[164]:


#AUC-ROC CURVE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# In[165]:


roc=roc_auc_score(y1_te,op3)


# In[166]:


fpr,tpr,threshold=roc_curve(y1_te,df1.iloc[:,-1])


# In[167]:


plt.plot(fpr,tpr)


# In[168]:


#PRC Curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score


# In[169]:


precision,recall,threshold=precision_recall_curve(y1_te,df1.iloc[:,-1])


# In[170]:


plt.plot(recall,precision,label="Precision-Recall")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="upper left")
plt.title("PRC-CURCE")


# # IMBLearn Oversampling

# In[171]:


#import imblearn


# In[172]:


#from imblearn.over_sampling import RandomOverSampler


# In[173]:


#cimb_0,cimb_1=data["income"].value_counts()
#cimb_0,cimb_1


# In[174]:


#trainimb_0=data[data["income"]==0]
#trainimb_1=data[data["income"]==1]
#trainimb_0.shape,trainimb_1.shape


# In[175]:


#trainimb_0.head(2)
#trainimb_1.head(2)


# In[176]:


#class_1imb=trainimb_1.sample(cimb_0,replace=True)


# In[177]:


#class_1imb.head()
#class_1imb.shape,trainimb_0.shape


# In[178]:


#class11_0=pd.concat([class_1imb,trainimb_0],axis=0)
#class11_0.shape


# In[179]:


#class11_0.income.value_counts()


# In[180]:


#class11_0.skew()


# In[181]:


'''import matplotlib.pyplot as plt
import seaborn as sns

num_cols = ['age', 'fnlwgt', 'hours_per_week']
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))

for i, col in enumerate(num_cols):
    sns.boxplot(class11_0[col], ax=axes[i])  # Plot boxplot for the current column
    axes[i].set_title(f'Boxplot for {col}')  # Set title for the current subplot

plt.tight_layout()
plt.show()'''


# In[182]:


# Outliers


# In[183]:


#sns.distplot(class11_0.age)


# In[184]:


#sns.boxplot(data=class11_0,y="age")


# In[185]:


#ub3=class11_0.age.mean()+3*class11_0.age.std()
#lb3=class11_0.age.mean()-3*class11_0.age.std()
#print(ub3,lb3)


# In[186]:


#class11_0.loc[class11_0["age"]>62.17673505682626,"age"]=62.17673505682626


# In[187]:


#sns.boxplot(data=class11_0,y="age")


# In[188]:


#sns.distplot(class11_0.age)


# In[189]:


#sns.distplot(class11_0.hours_per_week)


# In[190]:


#sns.boxplot(data=class11_0,y="hours_per_week")


# In[191]:


#ub4=class11_0.hours_per_week.mean()+3*class11_0.hours_per_week.std()
#lb4=class11_0.hours_per_week.mean()-3*class11_0.hours_per_week.std()
#print(ub4,lb4)


# In[192]:


#class11_0.loc[class11_0["hours_per_week"]>76.83375091868467,"hours_per_week"]=76.83375091868467


# In[193]:


#sns.boxplot(data=class11_0,y="hours_per_week")


# In[194]:


#sns.distplot(class11_0.hours_per_week)


# In[195]:


#skewness treatment


# In[196]:


#class11_0.skew()


# In[197]:


#sns.distplot(class11_0.fnlwgt)#iqr


# In[198]:


#class11_0.fnlwgt.hist()


# In[199]:


#class11_0.fnlwgt.describe()


# In[200]:


#IQR2=class11_0.fnlwgt.quantile(0.75)-class11_0.fnlwgt.quantile(0.25)
#IQR2


# In[201]:


#ub5=class11_0.fnlwgt.quantile(0.75)+1.5*IQR2
#lb5=class11_0.fnlwgt.quantile(0.25)-1.5*IQR2
#print(ub2,lb2)


# In[202]:


#class11_0.loc[class11_0["fnlwgt"]>409613.375,"fnlwgt"]=409613.375


# In[203]:


#sns.distplot(class11_0.fnlwgt)


# In[204]:


#class11_0.fnlwgt.hist()


# In[205]:


#class11_0.skew()


# In[206]:


#d_x1=data.iloc[:,1:-1]
#d_y1=data.iloc[:,-1]


# In[207]:


#from imblearn.over_sampling import RandomOverSampler


# In[208]:


#a1=RandomOverSampler(random_state=42)


# In[209]:


#x2,y2=a1.fit_resample(d_x1,d_y1)
#x2.shape,y2.shape


# In[210]:


#x_tr2,x_te2,y_tr2,y_te2=train_test_split(x2,y2,test_size=0.2,random_state=100)
#x_tr2.shape,x_te2.shape,y_tr2.shape,y_te2.shape


# # Logistic regression - IMBLearn Oversampling

# In[211]:


#lg.fit(x_tr2,y_tr2)


# In[212]:


#lgm1=lg.predict(x_te2)


# In[213]:


#lgmc1=confusion_matrix(y_te2,lgm1)
#lgmc1


# In[214]:


#lgma1=lgmc1.diagonal().sum()/lgmc1.sum()*100
#lgma1


# In[215]:


#print(classification_report(y_te2,lgm1))


# In[216]:


#lg4_precision=precision_score(y_te2,lgm1)
#lg4_recall=recall_score(y_te2,lgm1)
#lg4_f1_score=f1_score(y_te2,lgm1)
#print(lg4_precision,lg4_recall,lg4_f1_score)


# # Decision Tree - IMBLearn Oversampling

# In[217]:


#dt.fit(x_tr2,y_tr2)


# In[218]:


#dtm1=dt.predict(x_te2)


# In[219]:


#dtmc1=confusion_matrix(y_te2,dtm1)
#dtmc1


# In[220]:


#dtma1=dtmc1.diagonal().sum()/dtmc1.sum()*100
#dtma1


# In[221]:


#print(classification_report(y_te2,dtm1))


# In[222]:


#dt4_precision=precision_score(y_te2,dtm1)
#dt4_recall=recall_score(y_te2,dtm1)
#dt4_f1_score=f1_score(y_te2,dtm1)
#print(dt4_precision,dt4_recall,dt4_f1_score)


# # Random Forest - IMBLearn Oversampling

# In[223]:


#rf.fit(x_tr2,y_tr2)


# In[224]:


#rfm1=rf.predict(x_te2)


# In[225]:


#df2=pd.DataFrame(rfm1)


# In[226]:


#rfmc1=confusion_matrix(y_te2,rfm1)
#rfmc1


# In[227]:


#rfma1=rfmc1.diagonal().sum()/rfmc1.sum()*100
#rfma1


# In[228]:


#print(classification_report(y_te2,rfm1))


# In[229]:


#rf4_precision=precision_score(y_te2,rfm1)
#rf4_recall=recall_score(y_te2,rfm1)
#rf4_f1_score=f1_score(y_te2,rfm1)
#print(rf4_precision,rf4_recall,rf4_f1_score)


# # XGB - IMBLearn Oversampling

# In[230]:


#xgb.fit(x_tr2,y_tr2)


# In[231]:


#xgbm1=xgb.predict(x_te2)


# In[232]:


#xgbmc1=confusion_matrix(y_te2,xgbm1)
#xgbmc1


# In[233]:


#xgbma1=xgbmc1.diagonal().sum()/xgbmc1.sum()*100
#xgbma1


# In[234]:


#print(classification_report(y_te2,xgbm1))


# In[235]:


#xgb4_precision=precision_score(y_te2,xgbm1)
#xgb4_recall=recall_score(y_te2,xgbm1)
#xgb4_f1_score=f1_score(y_te2,xgbm1)
#Zprint(xgb4_precision,xgb4_recall,xgb4_f1_score)


# # KNN - IMBLearn Oversampling

# In[236]:


#knn.fit(x_tr2,y_tr2)


# In[237]:


#knnm1=knn.predict(x_te2)


# In[238]:


#knnmc1=confusion_matrix(y_te2,knnm1)
#knnmc1


# In[239]:


#knnma1=knnmc1.diagonal().sum()/knnmc1.sum()*100
#knnma1


# In[240]:


#print(classification_report(y_te2,knnm1))


# In[241]:


#knn4_precision=precision_score(y_te2,knnm1)
#knn4_recall=recall_score(y_te2,knnm1)
#knn4_f1_score=f1_score(y_te2,knnm1)
#print(knn4_precision,knn4_recall,knn4_f1_score)


# # NB - IMBLearn Oversampling

# In[242]:


#nb.fit(x_tr2,y_tr2)


# In[243]:


#nbm1=nb.predict(x_te2)


# In[244]:


#nbmc1=confusion_matrix(y_te2,nbm1)
#nbmc1


# In[245]:


#nbma1=nbmc1.diagonal().sum()/nbmc1.sum()*100
#nbma1


# In[246]:


#print(classification_report(y_te2,nbm1))


# In[247]:


#nb4_precision=precision_score(y_te2,nbm1)
#nb4_recall=recall_score(y_te2,nbm1)
#nb4_f1_score=f1_score(y_te2,nbm1)
#print(nb4_precision,nb4_recall,nb4_f1_score)


# # SVM - IMBLearn Oversampling

# In[248]:


#svm.fit(x_tr2,y_tr2)


# In[249]:


#svmm1=svm.predict(x_te2)


# In[250]:


#svmmc1=confusion_matrix(y_te2,svmm1)
#svmmc1


# In[251]:


#svmma1=svmmc1.diagonal().sum()/svmmc1.sum()*100
#svmma1


# In[252]:


#print(classification_report(y_te2,svmm1))


# In[253]:


#svm4_precision=precision_score(y_te2,svmm1)
#svm4_recall=recall_score(y_te2,svmm1)
#svm4_f1_score=f1_score(y_te2,svmm1)
#print(svm4_precision,svm4_recall,svm4_f1_score)


# In[254]:


#LO11=["Logistic","DT","RF","XGB","KNN","NB","SVM"]
#LO22=[lgma1,dtma1,rfma1,xgbma1,knnma1,nbma1,svmma1]
#LO33=[lg4_precision,dt4_precision,rf4_precision,xgb4_precision,knn4_precision,nb4_precision,svm4_precision]
#LO44=[lg4_recall,dt4_recall,rf4_recall,xgb4_recall,knn4_recall,nb4_recall,svm4_recall]
#LO55=[lg4_f1_score,dt4_f1_score,rf4_f1_score,xgb4_f1_score,knn4_f1_score,nb4_f1_score,svm4_f1_score]


# In[255]:


#final_df2=pd.DataFrame({"Model":LO11,"Accuracy":LO22,"Precision":LO33,"Recall":LO44,"F1_score":LO55})
#final_df2


# In[256]:


#Confusion matrix for RF model because the performance of the model is good as compared to other models
#cm2=confusion_matrix(y_te2,rfm1)
#sns.heatmap(cm2,annot=True,fmt="d",cmap="Spectral")
#plt.title("Confusion Matrix")
#plt.show()


# In[257]:


#AUC-ROC CURVE
#from sklearn.metrics import roc_auc_score
#from sklearn.metrics import roc_curve


# In[258]:


#roc=roc_auc_score(y_te2,rfm1)


# In[259]:


#fpr,tpr,threshold=roc_curve(y_te2,df2.iloc[:,-1])


# In[260]:


#plt.plot(fpr,tpr)


# In[261]:


#PRC Curve
#from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import f1_score


# In[262]:


#precision,recall,threshold=precision_recall_curve(y_te2,df2.iloc[:,-1])


# In[263]:


#plt.plot(recall,precision,label="Precision-Recall")
#plt.xlabel("Recall")
#plt.ylabel("Precision")
#plt.legend(loc="upper left")
#plt.title("PRC-CURCE")


# In[264]:


print("               *****Base Model****")
print(final_df)
print("\n")

print("\n          *****Random Oversampling Model*****")
print(final_df1)
print("\n")


# In[265]:


#Conclusion:
    #OverSampling - Random: RF highest Accuracy with high TN
    #               IMBlearn: RF has highest Acccuracy with high TN


# In[266]:


d1=data.corr()
d1


# In[267]:


#Visualisation for correlation
plt.figure(figsize=(15,10))
heatmap=sns.heatmap(d1,linewidth=1,annot=True,cmap=plt.cm.Blues)
plt.title("Heatmap using seaborn")
plt.show()


# In[268]:


x=data.sample(frac=0.05,replace=True,random_state=1)
plt.figure(figsize=(5,3))
sns.barplot(x="income",y="occupation",data=x)


# In[269]:


#Feature Selection


# In[270]:


rf.feature_importances_


# In[271]:


rf1=pd.DataFrame(rf.feature_importances_)
rf1


# In[272]:


r=pd.DataFrame({"Feature_score":list(rf.feature_importances_),"columns":list(x_train.iloc[:,:-1].columns)})
r


# In[273]:


from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()


# In[274]:


rfe=RFE(dtree,n_features_to_select=5)


# In[275]:


rfe.fit(data.iloc[:,:-1],data.iloc[:,-1])


# In[276]:


f1=pd.DataFrame({"feature":list(rfe.support_),"column":list(data.iloc[:,:-1].columns)})


# In[277]:


rfe.support_


# In[278]:


f1


# In[279]:


data2=data.drop(["workclass","education","marital_status","occupation","race","sex","capital_loss",
                 "hours_per_week","native_country"],axis=1)
data2


# In[280]:


d_x2=data2.iloc[:,1:-1]
d_y2=data2.iloc[:,-1]


# In[281]:


x_tr3,x_te3,y_tr3,y_te3=train_test_split(d_x2,d_y2,test_size=0.2,random_state=100)
x_tr3.shape,x_te3.shape,y_tr3.shape,y_te3.shape


# # Logistic Regression

# In[282]:


lg.fit(x_tr3,y_tr3)


# In[283]:


lg_p_re=lg.predict(x_te3)


# In[284]:


c1_re=confusion_matrix(y_te3,lg_p_re)
c1_re


# In[285]:


a1_re=c1_re.diagonal().sum()/c1_re.sum()*100
a1_re


# In[286]:


print(classification_report(y_te3,lg_p_re))


# In[287]:


lg5_precision=precision_score(y_te3,lg_p_re)
lg5_recall=recall_score(y_te3,lg_p_re)
lg5_f1_score=f1_score(y_te3,lg_p_re)
print(lg5_precision,lg5_recall,lg5_f1_score)


# # DT

# In[288]:


dt.fit(x_tr3,y_tr3)


# In[289]:


dt_p_re=dt.predict(x_te3)


# In[290]:


c2_re=confusion_matrix(y_te3,dt_p_re)
c2_re


# In[291]:


a2_re=c2_re.diagonal().sum()/c2_re.sum()*100
a2_re


# In[292]:


print(classification_report(y_te3,dt_p_re))


# In[293]:


dt5_precision=precision_score(y_te3,dt_p_re)
dt5_recall=recall_score(y_te3,dt_p_re)
dt5_f1_score=f1_score(y_te3,dt_p_re)
print(dt5_precision,dt5_recall,dt5_f1_score)


# # RF

# In[294]:


rf.fit(x_tr3,y_tr3)


# In[295]:


rf_p_re=rf.predict(x_te3)


# In[296]:


c3_re=confusion_matrix(y_te3,rf_p_re)
c3_re


# In[297]:


a3_re=c3_re.diagonal().sum()/c3_re.sum()*100
a3_re


# In[298]:


print(classification_report(y_te3,rf_p_re))


# In[299]:


rf5_precision=precision_score(y_te3,rf_p_re)
rf5_recall=recall_score(y_te3,rf_p_re)
rf5_f1_score=f1_score(y_te3,rf_p_re)
print(rf5_precision,rf5_recall,rf5_f1_score)


# # KNN

# In[300]:


knn.fit(x_tr3,y_tr3)


# In[301]:


knn_p_re=knn.predict(x_te3)


# In[302]:


c4_re=confusion_matrix(y_te3,knn_p_re)
c4_re


# In[303]:


a4_re=c4_re.diagonal().sum()/c4_re.sum()*100
a4_re


# In[304]:


print(classification_report(y_te3,knn_p_re))


# In[305]:


knn5_precision=precision_score(y_te3,knn_p_re)
knn5_recall=recall_score(y_te3,knn_p_re)
knn5_f1_score=f1_score(y_te3,knn_p_re)
print(knn5_precision,knn5_recall,knn5_f1_score)


# # XGB

# In[306]:


xgb.fit(x_tr3,y_tr3)


# In[307]:


xgb_p_re=xgb.predict(x_te3)


# In[308]:


df3=pd.DataFrame(x_te3)


# In[309]:


c5_re=confusion_matrix(y_te3,xgb_p_re)
c5_re


# In[310]:


a5_re=c5_re.diagonal().sum()/c5_re.sum()*100
a5_re


# In[311]:


print(classification_report(y_te3,xgb_p_re))


# In[312]:


xgb5_precision=precision_score(y_te3,xgb_p_re)
xgb5_recall=recall_score(y_te3,xgb_p_re)
xgb5_f1_score=f1_score(y_te3,xgb_p_re)
print(xgb5_precision,xgb5_recall,xgb5_f1_score)


# # Naive Bayes

# In[313]:


nb.fit(x_tr3,y_tr3)


# In[314]:


nb_p_re=nb.predict(x_te3)


# In[315]:


c6_re=confusion_matrix(y_te3,nb_p_re)
c6_re


# In[316]:


a6_re=c6_re.diagonal().sum()/c6_re.sum()*100
a6_re


# In[317]:


print(classification_report(y_te3,nb_p_re))


# In[318]:


nb5_precision=precision_score(y_te3,nb_p_re)
nb5_recall=recall_score(y_te3,nb_p_re)
nb5_f1_score=f1_score(y_te3,nb_p_re)
print(nb5_precision,nb5_recall,nb5_f1_score)


# # SVM

# In[319]:


svm.fit(x_tr3,y_tr3)


# In[320]:


svm_p_re=svm.predict(x_te3)


# In[321]:


c7_re=confusion_matrix(y_te3,svm_p_re)
c7_re


# In[322]:


a7_re=c7_re.diagonal().sum()/c7_re.sum()*100
a7_re


# In[323]:


print(classification_report(y_te3,svm_p_re))


# In[324]:


svm5_precision=precision_score(y_te3,svm_p_re)
svm5_recall=recall_score(y_te3,svm_p_re)
svm5_f1_score=f1_score(y_te3,svm_p_re)
print(svm5_precision,svm5_recall,svm5_f1_score)


# In[325]:


LRE1=["Logistic","DT","RF","KNN","XGB","NB","SVM"]
LRE2=[a1_re,a2_re,a3_re,a4_re,a5_re,a6_re,a7_re]
LRE3=[lg5_precision,dt5_precision,rf5_precision,knn5_precision,xgb5_precision,nb5_precision,svm5_precision]
LRE4=[lg5_recall,dt5_recall,rf5_recall,knn5_recall,xgb5_recall,nb5_recall,svm5_recall]
LRE5=[lg5_f1_score,dt5_f1_score,rf5_f1_score,knn5_f1_score,xgb5_f1_score,nb5_f1_score,svm5_f1_score]


# In[326]:


final_df3=pd.DataFrame({"Model":LRE1,"Accuracy":LRE2,"Precision":LRE3,"Recall":LRE4,"F1_score":LRE5})
final_df3


# In[327]:


#Conclusion : After applying the feature selection technique the accuracy of the model is quiet decreases.
#In this case, XGB is good fitted to the data and also we can conclude the result by using roc-auc curve.


# In[328]:


#Confusion matrix for XGB model because the performance of the model is good as compared to other models
cm2=confusion_matrix(y_te3,xgb_p_re)
sns.heatmap(cm2,annot=True,fmt="d",cmap="Spectral")
plt.title("Confusion Matrix")
plt.show()


# In[329]:


#AUC-ROC CURVE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# In[330]:


roc=roc_auc_score(y_te3,xgb_p_re)


# In[331]:


fpr,tpr,threshold=roc_curve(y_te3,df3.iloc[:,-1])


# In[332]:


plt.plot(fpr,tpr)


# In[333]:


#PRC Curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score


# In[334]:


precision,recall,threshold=precision_recall_curve(y_te3,df3.iloc[:,-1])


# In[335]:


plt.plot(recall,precision,label="Precision-Recall")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="upper left")
plt.title("PRC-CURCE")


# In[336]:


print("               *****Base Model****")
print(final_df)
print("\n")

print("\n          *****Random Oversampling Model*****")
print(final_df1)
print("\n")

print("\n          *****Feature Selection Model*****")
print(final_df3)


# In[337]:


#feature selection is not suitable for this dataset because 
#before feature selection model is giving good results as compared to feature selection


# In[ ]:




