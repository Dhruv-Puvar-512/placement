import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset from a CSV file
data = pd.read_csv("Placement_Data_Full_Class.csv")
# Convert NaN values in the "salary" column to 0
data["salary"].fillna(0, inplace=True)


# Preprocess the data
# Encode categorical variables
label_encoders = {}
categorical_columns = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation", "status"]

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Select features and labels
features = data[["ssc_p", "hsc_p", "degree_p", "workex", "specialisation", "mba_p"]]
labels = data["status"]



# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Build a Random Forest classifier (you can use other algorithms)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)



import pickle

# Save the model to a .pkl file using pickle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Predict whether a new candidate will be placed or not
new_candidate_data = pd.DataFrame({
    "ssc_p": [64.0],
    "hsc_p": [73.5],
    "degree_p": [73.0],
    "workex": [0],  # 1 for "Yes," 0 for "No"
    "specialisation": [0],  # 0 for "Mkt&HR," 1 for "Mkt&Fin"
    "mba_p": [56.70]
})

predicted_status = clf.predict(new_candidate_data)
print(f"Predicted Status for the new candidate: {'Placed' if predicted_status[0] == 1 else 'Not Placed'}")