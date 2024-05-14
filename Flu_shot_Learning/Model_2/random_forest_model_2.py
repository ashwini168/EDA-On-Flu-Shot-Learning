import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Load the datasets
train_features = pd.read_csv("training_set_features.csv")
train_labels = pd.read_csv("training_set_labels.csv")
test_features = pd.read_csv("test_set_features.csv")


# Split features and target variables
X_train = train_features.drop(columns=['respondent_id'])
y_train_h1n1 = train_labels['h1n1_vaccine']
y_train_seasonal = train_labels['seasonal_vaccine']
X_test = test_features.drop(columns=['respondent_id'])

# Preprocessing pipeline
numeric_features = X_train.select_dtypes(include=['int64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Append classifier to preprocessing pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])

# Train the model for H1N1 vaccine
clf.fit(X_train, y_train_h1n1)

# Predict probabilities for H1N1 vaccine
y_pred_proba_h1n1 = clf.predict_proba(X_test)[:, 1]  # Probability of class 1 (vaccine received)
y_pred_proba_h1n1_rounded = [round(prob, 1) for prob in y_pred_proba_h1n1]  # Round to 1 decimal place

# Train the model for seasonal vaccine
clf.fit(X_train, y_train_seasonal)

# Predict probabilities for seasonal vaccine
y_pred_proba_seasonal = clf.predict_proba(X_test)[:, 1]  # Probability of class 1 (vaccine received)
y_pred_proba_seasonal_rounded = [round(prob, 1) for prob in y_pred_proba_seasonal]  # Round to 1 decimal place

# Save the probabilities
predictions_df = pd.DataFrame({'respondent_id': test_features['respondent_id'],
                               'h1n1_vaccine': y_pred_proba_h1n1_rounded,
                               'seasonal_vaccine': y_pred_proba_seasonal_rounded})
predictions_df.to_csv('predictionsmodel_2.csv', index=False)
