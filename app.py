import streamlit as st
import pickle

# Load the saved model using pickle
with open('model.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

# Streamlit app
st.title("Candidate Placement Prediction")

# Input fields for candidate details
ssc_p = st.number_input("SSC Percentage", min_value=0.0, max_value=100.0)
hsc_p = st.number_input("HSC Percentage", min_value=0.0, max_value=100.0)
degree_p = st.number_input("Degree Percentage", min_value=0.0, max_value=100.0)
workex = st.selectbox("Work Experience", ["No", "Yes"])
specialisation = st.selectbox("Specialization", ["Mkt&HR", "Mkt&Fin"])
mba_p = st.number_input("MBA Percentage", min_value=0.0, max_value=100.0)

# Predict
if st.button("Predict Placement"):
    # Prepare input data as a dictionary
    new_candidate_data = {
        "ssc_p": ssc_p,
        "hsc_p": hsc_p,
        "degree_p": degree_p,
        "workex": 1 if workex == "Yes" else 0,
        "specialisation": 1 if specialisation == "Mkt&Fin" else 0,
        "mba_p": mba_p
    }

    # Make predictions
    input_data = [list(new_candidate_data.values())]
    predicted_status = clf.predict(input_data)
    result = "Placed" if predicted_status[0] == 1 else "Not Placed"
    st.write(f"Predicted Status: {result}")
