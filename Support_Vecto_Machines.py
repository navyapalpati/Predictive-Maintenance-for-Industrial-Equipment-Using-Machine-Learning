import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('predictive_maintenance.csv')

# Drop unnecessary columns for training the model (UDI, Product ID, Type)
df = df.drop(['UDI', 'Product ID', 'Type'], axis=1)

# Convert categorical target variable 'FailureType' to numerical
le = LabelEncoder()
df['FailureType'] = le.fit_transform(df['FailureType'])

# Split the data into features (X) and target variable (y)
X = df.drop(['Target', 'FailureType'], axis=1)
y = df['FailureType']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (important for SVMs)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create an SVM classifier
svm_model = SVC(kernel='linear', C=1.0, random_state=42)

# Train the model
svm_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('\nClassification Report:\n', classification_rep)
