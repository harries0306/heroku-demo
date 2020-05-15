import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as lib
import pickle


df=pd.read_csv('advertising.csv')

df.describe()

#df.Age.hist(bins=40,grid=False)

from sklearn.model_selection import train_test_split

X = df[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = df['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


predictions = logmodel.predict(X_test)

pickle.dump(logmodel, open('model.pkl','wb'))


model = pickle.load(open('model.pkl','rb'))
print(model.predict([[75,30,700000,250,0]]))