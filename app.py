import streamlit as st
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open("RF.pkl", "rb"))

sexs = ["female", "male"]
smokers = ["no", "yes"]
regions = ["northeast", "northwest", "southeast", "southwest"]

st.title("Medical Cost Prediction")
age = st.number_input("Age")
sex = st.selectbox("Sex", sexs)
bmi = st.number_input("BMI")
children = st.number_input("Children")
smoker = st.selectbox("Smoker", smokers)
region = st.selectbox("Region", regions)

if st.button("Predict"):
	sex = sexs.index(sex)
	smoker = smokers.index(smoker)
	region = regions.index(region)

	test = np.array([[age, sex, bmi, children, smoker, region]])
	res = model.predict(test).item()
	st.success("Predicted Cost:" + str(res))
