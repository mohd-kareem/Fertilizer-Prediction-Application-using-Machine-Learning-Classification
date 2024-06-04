import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Fertilizer Recommendation.csv")
df.head()

#changing values to numeric form
le = LabelEncoder()
df['Soil Type']=le.fit_transform(df['Soil Type'])
df['Crop Type'] = le.fit_transform(df['Crop Type'])
df['Fertilizer Name'] = le.fit_transform(df['Fertilizer Name'])

#x = df.drop(columns='Fertilizer Name')
#y = df['Fertilizer Name']
x = df.iloc[:,:7]
y = df.iloc[:,8]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error

kn = KNeighborsClassifier(n_neighbors=5)
s = SVC()
rf = RandomForestClassifier()
dt = DecisionTreeClassifier()

rf.fit(x_train,y_train)
dt.fit(x_train,y_train)
s.fit(x_train,y_train)
kn.fit(x_train,y_train)

st.header("Fertilizer Prediction")
st.sidebar.header("User Input")
temp = st.sidebar.slider("Temperature",0,70,24)
humi = st.sidebar.slider("Humididty",0,80,24)
soil = st.sidebar.slider("Soil Type",0,12,2)
crop = st.sidebar.slider("Crop Type",0,11,2)
nitr = st.sidebar.slider("Nitrogen",0,70,24)
pota = st.sidebar.slider("Postassium",0,11,3)
phos = st.sidebar.slider("Phosphorus",0,50,24)



soil_enc = 0
if soil == "Sandy":
    soil_enc = 4
elif soil == "Loamy":
    soil_enc = 2
elif soil == "Black":
    soil_enc = 0
elif soil == "Red":
    soil_enc = 3
elif soil == "Clayey":
    soil_enc = 1

crop_enc = 0
if crop == "Maize":
    crop_enc = 3
elif crop == "Sugar":
    crop_enc = 8
elif crop == "Cotton":
    crop_enc = 1
elif crop == "Tobacco":
    crop_enc = 9
elif crop == "Paddy":
    crop_enc = 6
elif crop == "Barley":
    crop_enc = 0
elif crop == "Wheat":
    crop_enc = 10
elif crop == "Millets":
    crop_enc = 4
elif crop == "Oil seeds":
    crop_enc = 5
elif crop == "Ground Nuts":
    crop_enc = 7
        

pred1 = rf.predict([[temp,humi,soil_enc,crop_enc,nitr,pota,phos]])
y_pred2 = rf.predict(x_test)

pred2 = dt.predict([[temp,humi,soil_enc,crop_enc,nitr,pota,phos]])
y_pred1 = dt.predict(x_test) #for the user input 
#it prints the output 

pred3 = s.predict([[temp,humi,soil_enc,crop_enc,nitr,pota,phos]])
y_pred3 = s.predict(x_test)

pred4 = kn.predict([[temp,humi,soil_enc,crop_enc,nitr,pota,phos]])
y_pred4 = kn.predict(x_test)


st.subheader("Predicted value of Fertilizer  by Decision Tree")
st.write(pred1)
st.subheader("Accuracy")
st.write(mean_absolute_error(y_test,y_pred1))

st.subheader("Predicted value of Fertilizer  by Random Forest")
st.write(pred2)
st.subheader("Accuracy")
st.write(mean_absolute_error(y_test,y_pred2))

st.subheader("Predicted value of Fertilizer  by Support Vector Classifier")
st.write(pred3)
st.subheader("Accuracy")
st.write(mean_absolute_error(y_test,y_pred3))

st.subheader("Predicted value of Fertilizer  by Kneighbour Classifier")
st.write(pred4)
st.subheader("Accuracy")
st.write(mean_absolute_error(y_test,y_pred4))
