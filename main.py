import joblib
import pandas as pd
import seaborn as sns
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression

# 1. Load dataset
df = sns.load_dataset('titanic')

df = df[["pclass", 'sex', 'age', 'fare', 'survived']]
x = df.drop('survived', axis=1)
y = df['survived']

# 2. Define clear list arrays for column names
numeric_cols = ['age', 'fare']
categorical_cols = ['sex', 'pclass']

# 3. Define the numeric processing pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 4. Define the categorical processing pipeline (Fixed spelling/definition)
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder())
])

# 5. Combine them using ColumnTransformer
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# 6. Final unified pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

# 7. Fit the model on the training data
pipeline.fit(x, y)

# 8. Create folder and save the pickled model
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/pipeline.pkl")

print("Pipeline saved Successfully!")