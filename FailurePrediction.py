# Import necessary libraries
import pandas as pd  # Import pandas with the 'pd' alias
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset
df = pd.read_csv('predictive_maintenance.csv')

# Drop columns that are not needed for prediction
df = df.drop(['UDI', 'Product ID', 'Type'], axis=1)

# Drop rows with missing values in the target variable
df = df.dropna(subset=['FailureType'])

# Convert categorical target variable to numerical
df['FailureType'] = df['FailureType'].map({'No Failure': 0, 'Failure': 1})

# Split data into features (X) and target variable (y)
X = df.drop('FailureType', axis=1)
y = df['FailureType']

# Check for NaN values in the target variable
if y.isnull().any():
    print("Warning: NaN values found in the target variable. Handling missing values.")
    y = y.dropna()

# Ensure the indices are aligned after handling missing values
X = X.loc[y.index]
# Alternatively, you can reset the indices:
# X = X.reset_index(drop=True)
# y = y.reset_index(drop=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the trained model for future use
joblib.dump(model, 'predictive_maintenance_model.joblib')

# Now, let's use the trained model for prediction and generate alerts
def predict_and_alert(new_data):
    # Load the trained model
    loaded_model = joblib.load('predictive_maintenance_model.joblib')

    # Make predictions for the new data
    predictions = loaded_model.predict(new_data)

    # Check for failures and generate alerts
    alerts = []
    for i, prediction in enumerate(predictions):
        if prediction == 1:
            alerts.append(f"Alert: Failure predicted for data point {i+1}")

    return alerts

# Example: Predict and generate alerts for new data
new_data = pd.DataFrame({
    'Airtemperature[K]': [298.0],
    'Processtemperature[K]': [308.5],
    'Rotationalspeed[rpm]': [1500],
    'Torque[Nm]': [45.0],
    'Toolwear[min]': [6]
})

alerts = predict_and_alert(new_data)
print(alerts)