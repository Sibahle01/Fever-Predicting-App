import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Load data
data_path = r"C:\Users\sbahl\OneDrive\Desktop\enhanced_fever_medicine_recommendation.csv"
data = pd.read_csv(data_path)

# First, let's analyze our categorical columns
print("Analyzing categorical columns...")
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    print(f"\nUnique values in {col}:")
    print(data[col].unique())

# Fill missing values
data['Previous_Medication'] = data['Previous_Medication'].fillna("None")

# Define target and features
target = 'Fever_Severity'
features = data.drop(columns=[target, 'Recommended_Medication'])
labels = data[target]

# Create dictionary to store encoders
encoders_dict = {}

# Handle categorical columns
for column in categorical_columns:
    if column not in [target, 'Recommended_Medication']:
        encoders_dict[column] = LabelEncoder()
        # Fit encoder on the actual values in the dataset
        encoders_dict[column].fit(features[column].unique())
        features[column] = encoders_dict[column].transform(features[column])
        # Print the classes for verification
        print(f"\nClasses for {column}:")
        print(encoders_dict[column].classes_)

# Scale numerical features
scaler = StandardScaler()
numerical_columns = features.select_dtypes(include=['float64', 'int64']).columns
features[numerical_columns] = scaler.fit_transform(features[numerical_columns])

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save all components
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(encoders_dict, file)

print("Files saved successfully!")

# Save the actual categories for reference
categories_info = {column: list(encoder.classes_) for column, encoder in encoders_dict.items()}
with open('categories_info.pkl', 'wb') as file:
    pickle.dump(categories_info, file)

print("\nCategories for each column:")
for column, categories in categories_info.items():
    print(f"\n{column}:")
    print(categories)
