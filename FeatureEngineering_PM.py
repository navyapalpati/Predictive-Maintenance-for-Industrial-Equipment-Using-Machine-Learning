import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the data
df = pd.read_csv("predictive_maintenance.csv")

# Drop irrelevant columns
df = df.drop(['UDI', 'Product ID', 'Type'], axis=1)

# Convert 'Failure Type' using label encoding
label_encoder = LabelEncoder()
df['Failure Type'] = label_encoder.fit_transform(df['Failure Type'])

# Feature Engineering
# Example: Creating a new feature 'Temperature Difference'
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

# Display the preprocessed and engineered data
print("X_train_scaled with engineered features:")
print(X_train_scaled[:5])

print("y_train:")
print(y_train[:5])