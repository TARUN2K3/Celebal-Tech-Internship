import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Data
data = pd.read_csv('titanic.csv')

# Handle Missing Values
imputer = SimpleImputer(strategy='mean')
data[['column_with_missing_values']] = imputer.fit_transform(data[['column_with_missing_values']])

# Encode Categorical Variables
categorical_cols = ['categorical_column1', 'categorical_column2']
encoder = OneHotEncoder(drop='first', sparse=False)
encoded_features = encoder.fit_transform(data[categorical_cols])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
data = data.join(encoded_df).drop(categorical_cols, axis=1)

# Feature Scaling
scaler = StandardScaler()
numerical_cols = ['numerical_column1', 'numerical_column2']
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Feature Engineering
data['new_feature'] = data['numerical_column1'] * data['numerical_column2']

# Feature Selection (Example: Select top 5 features based on RandomForest importance)
X = data.drop('target', axis=1)
y = data['target']

# Split data for training feature selection model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a model to determine feature importance
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Get feature importances and select top features
importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = importances.nlargest(5).index

# Create final dataset with selected features
final_data = data[top_features]

# Train-test split
X_final_train, X_final_test, y_final_train, y_final_test = train_test_split(final_data, y, test_size=0.3, random_state=42)

# Train a final model on selected features
final_model = RandomForestClassifier()
final_model.fit(X_final_train, y_final_train)

# Predict and evaluate
y_pred = final_model.predict(X_final_test)
accuracy = accuracy_score(y_final_test, y_pred)
print(f"Model Accuracy: {accuracy}")
