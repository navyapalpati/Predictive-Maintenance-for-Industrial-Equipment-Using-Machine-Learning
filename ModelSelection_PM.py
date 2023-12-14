import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the data
df = pd.read_csv("predictive_maintenance.csv")

# Drop irrelevant columns
df = df.drop(['UDI', 'Product ID', 'Type'], axis=1)

# Convert 'Failure Type' using label encoding
label_encoder = LabelEncoder()
df['Failure Type'] = label_encoder.fit_transform(df['Failure Type'])

# Feature Engineering
df['Temperature Difference'] = df['Air temperature [K]'] - df['Process temperature [K]']

# Split the data into features (X) and target variable (y)
X = df.drop('Failure Type', axis=1)
y = df['Failure Type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Choose a Model (Example: Decision Tree Classifier)
model = DecisionTreeClassifier(random_state=42)

# Train the Model
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
