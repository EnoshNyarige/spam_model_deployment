import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


st.title("The Email Classification Predictor")
st.subheader("Determine if that email you received is spam or not")
html_temp = """
	<div style="background-color:black;padding:10px">
	<h3 style="color:white;text-align:center;">Machine Learning</h3>
	</div>
	<div>
	<h3 style="color:black;text-align:center;">Members</h3>
	<p style="color:black;text-align:center;">Deogratius Amani</p> 
	<p style="color:black;text-align:center;">Yves Mugenga</p> 
	<p style="color:black;text-align:center;">Enosh Nyarige</p>
	
	"""
st.markdown(html_temp,unsafe_allow_html=True)

user_input =st.text_area("Copy your email here","")
st.write("***The email provided reads;***\n\n" , user_input)

tokenizer_file = "vec_tokenizer.sav"
tokenizer = pickle.load(open(tokenizer_file, "rb"))


user_input = [user_input.split(" ")]
text_seq = tokenizer.texts_to_sequences(user_input)
padded_text_seq = pad_sequences(text_seq, maxlen=6, padding="post") 

# 
model_file = "spam_model.h5"
bilstm_model = load_model(model_file, compile = False)

y_pred = bilstm_model.predict(padded_text_seq)
y_pred = np.argmax(y_pred, axis=1)

if st.button("Predict"):
    if y_pred[0] == 0:
        st.write("The classifier says this a **Ham** email. No need for more caution")
    elif y_pred[0] == 1:
        st.write("The classifier says this a **Spam** email. Proceed with caution")
