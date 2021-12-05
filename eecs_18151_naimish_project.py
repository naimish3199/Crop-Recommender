import numpy as np
import pandas as pd
import pickle

data = pd.read_csv("C:/Users/naimi/OneDrive/Desktop/Python Final Project/Crop_recommendation.csv")

#Train Test Split

X = data[['N','P','K','temperature','humidity','ph','rainfall']]
Y = data['label']

"""## Naive Bayes"""

from sklearn.naive_bayes import GaussianNB

GNB = GaussianNB()
# training
GNB.fit(X,Y)

pickle.dump(GNB,open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[56, 69, 53,25.4,82,7,200]]))
