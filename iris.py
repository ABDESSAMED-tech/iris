import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.neighbors import KNeighborsClassifier as knn

st.title('Application of ML (RandomForestClassifier ) with Iris dataset ')
st.sidebar.header("Input parameter ")



def param():
    spL = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.3)
    spW = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.3)
    pL = st.sidebar.slider('Petal length', 1.0, 6.9, 5.3)
    pW = st.sidebar.slider('Petal Width', 1.0, 2.5, 1.3)
    data = {'Sepal length': spL, 'Sepal Width': spW,
            'Petal length': pL, 'petal Width': pW}
    fleure_params = pd.DataFrame(data,index=[0])
    return fleure_params

st.write('### we need to get category of flower  ')
df=param()
st.write(df)
iris=datasets.load_iris()
clf=rfc()
clf.fit(iris.data,iris.target)

prediction=clf.predict(df)
st.write("la catégorie de la fleur d'Iris est :")
st.write(iris.target_names[prediction])
