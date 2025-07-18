
import streamlit as st
import pickle
import pandas as pd

# Load your trained model pipeline
with open('xgb_prediction.pkl', 'rb') as f:
    model_deploy = pickle.load(f)

st.title("Product Return Predictor")

# Input fields
name = st.selectbox("Product Name:", ['Kindle', 'Fire HD', 'Digital Clock', 'Bluetooh Speaker', 'Travel Adapter'])
category = st.selectbox("Category:", ['e-Reader', 'Electronic', 'Tablet', 'Accessories'])
review = st.text_area("Customer Review")
rating = st.slider("Rating (1 to 5)", 1, 5, 3)
recommend = st.radio("Select Yes = 1, No = 0", ['1', '0'])


# Predict button
if st.button("Predict Recommendation"):
    # Build DataFrame input
    user_input = pd.DataFrame([{
        'name': name,
        'category': category,
        'review': review,
        'rating': rating,
        'recommend': recommend,
    }])
    user_input['text'] = user_input['name'] + ' ' + user_input['category'] + ' ' + user_input['review'] + ' ' + user_input['rating'] + ' ' + user_input['recommend']

    # Predict
    prediction = model_deploy.predict(user_input)[0]

    # Show result
    st.success("Good Product: 👍 YES" if prediction == 1 else "I'll Return the Products: 👎 NO")
