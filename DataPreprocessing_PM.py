import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the data
df = pd.read_csv("predictive_maintenance.csv")

# Display the first few rows of the dataframe
print(df.head())

# Drop irrelevant columns (if any)
# In this case, 'UDI', 'Product ID', and 'Type' might not be necessary for predictive maintenance.
df = df.drop(['UDI', 'Product ID', 'Type'], axis=1)

# Handle categorical variables using one-hot encoding or label encoding
# In this case, 'Failure Type' is a categorical variable
label_encoder = LabelEncoder()
df['Failure Type'] = label_encoder.fit_transform(df['Failure Type'])

# Split the data into features (X) and target variable (y)
X = df.drop('Failure Type', axis=1)
y = df['Failure Type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features (optional but often beneficial)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display the preprocessed data
print("X_train_scaled:")
print(X_train_scaled[:5])

print("y_train:")
print(y_train[:5])