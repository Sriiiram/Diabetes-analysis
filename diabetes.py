# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 12:28:39 2019

@author: Sriram
"""

# Replacing missing values with median
import pandas as pd
import numpy as np
data=pd.read_csv("project.csv")
data.fillna(data.median(), inplace=True)
data
data.isna().sum()
nums=data.dtypes[data.dtypes!='object'].index
nums

#finding skewness and removing it
skws=data[nums].skew().sort_values(ascending=False)
skewness=pd.DataFrame({'skew':skws})
skewness

#assigning the skewed values to new variables
from scipy.stats import boxcox
from scipy import stats
dft_insulin = stats.boxcox(data['Insulin'])[0]
dft_BMI=stats.boxcox(data['BMI'])[0]
dft_diabetes=stats.boxcox(data['DiabetesPedigreeFunction'])[0]
data['Insulin']=dft_insulin
data['BMI']=dft_BMI
data['DiabetesPedigreeFunction']=dft_diabetes
data

#boxplotting the columns
import matplotlib.pyplot as plt
import numpy as np
boxplot = data.boxplot(column=['Insulin', 'BMI', 'DiabetesPedigreeFunction'])

from sklearn.preprocessing import StandardScaler
features = ['Age', 'Pregnancies', 'SkinThickness', 'Glucose','BloodPressure','Insulin','DiabetesPedigreeFunction','BMI']
x = data.loc[:, features].values
y = data.loc[:,['Outcome']].values
x = StandardScaler().fit_transform(x)
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3'])
principalDf
import numpy as np
mean_vec = np.mean(x, axis=0)
cov_mat = (x- mean_vec).T.dot((x - mean_vec)) / (x.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)
finalDf = pd.concat([principalDf, data[['Outcome']]], axis = 1)
finalDf
pca.explained_variance_ratio_
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
data['Outcome'].value_counts()
import seaborn as sns
import matplotlib.pyplot as plt 
sns.countplot(x='Outcome',data=data,palette='hls')
plt.show()
plt.savefig('Count_plot')
from imblearn.over_sampling import SMOTE

tester=['Pregnancies']
X = data.loc[:,tester].values
y = data.loc[:,['Outcome']].values
resampling=SMOTE(sampling_strategy='auto')
xx,yy=resampling.fit_sample(X,y.ravel())
X_train, X_test, y_train, y_test = train_test_split(xx,yy, test_size=0.3) # 70% training and 30% test
reg = LogisticRegression(random_state=0)
reg.fit(X_train,y_train)
ppp1=reg.predict(X_test)
confusion_matrix(y_test,ppp1)
accuracy_score(y_test,ppp1)
recall_score(y_test,ppp1)
precision_score(y_test,ppp1)
classification_report(y_test,ppp1)
pd.DataFrame(yy)[0].value_counts().plot(kind='bar')


z = data.loc[:,features]
a = data.loc[:,['Outcome']].values
resampling=SMOTE(sampling_strategy='auto')
qq,ww=resampling.fit_sample(z,a.ravel())
X_train, X_test, y_train, y_test = train_test_split(qq,ww, test_size=0.3) # 70% training and 30% test
query2=LogisticRegression(random_state=42)
query2.fit(X_train,y_train)
ppp=query2.predict(X_test)
confusion_matrix(y_test,ppp)
accuracy_score(y_test,ppp)
classification_report(y_test,ppp)

recall_score(y_test,ppp)


from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score,mean_squared_error,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(qq, ww, test_size=0.3) # 70% training and 30% test
clf=RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
confusion_matrix(y_test,y_pred)
print("Accuracy:",accuracy_score(y_test, y_pred))
recall_score(y_test,y_pred)
classification_report(y_test,y_pred)

feature_imp = pd.Series(clf.feature_importances_,index=features).sort_values(ascending=False)
feature_imp
sns.barplot(x=feature_imp, y=feature_imp.index)
