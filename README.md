## Regression-with-an-Abalone-Dataset

### Dataset Description
The dataset for this competition (both train and test) was generated from a deep learning model trained on the **Abalone dataset**. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

### Files
- `train.csv`: the training dataset; `Rings` is the integer target
- `test.csv`: the test dataset; your objective is to predict the value of `Rings` for each row
- `sample_submission.csv`: a sample submission file in the correct format


## Python code 
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dropout, Dense
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from tensorflow.keras.callbacks import EarlyStopping
```

```python
# Reading the training data
df_0 = pd.read_csv("train.csv")
df = df_0.copy()
```

```python
# Reading the test data
test = pd.read_csv("test.csv")
df_test = test.copy()
```

```python
# Encoding categorical variable 'Sex' using LabelEncoder
label_encoder = LabelEncoder()
df['Sex_encoded'] = label_encoder.fit_transform(df['Sex'])
df_test['Sex_encoded'] = label_encoder.transform(df_test['Sex'])
```

```python
# Dropping the original 'Sex' column
df.drop(columns=['Sex'], inplace=True)
df_test.drop(columns=['Sex'], inplace=True)
```

```python
# Splitting the data into features (X) and target (y)
X = df.drop(columns=['Survived'])
y = df['Survived']
```

```python
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
```

```python
# Scaling the features using MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

```python
# Building the neural network model
model = Sequential()
model.add(Dense(units=256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(units=256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(units=64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=16, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=8, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=3, activation='relu'))

# Output layer
model.add(Dense(units=1, activation='linear'))
```

```python
# Compiling the model
model.compile(optimizer='adam', loss=MeanSquaredLogarithmicError(), metrics=['msle'])
```

```python
# Train the model
model.fit(x=X_train, y=y_train, validation_split=0.1, batch_size=512, epochs=1000, callbacks=[EarlyStopping(monitor="val_loss", mode="auto", verbose=1, patience=20)])
```

```python
# Function to evaluate metrics
def eval_metric(actual, pred):
    mae = mean_absolute_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    msle = mean_squared_log_error(actual, pred)
    rmse = np.sqrt(mse)
    score = r2_score(actual, pred)
    return print("r2_score:", score, "\nmae:", mae, "\nmse:", mse, "\nrmse:", rmse, "\nmsle:", msle)
```

```python
# Predict on test data and evaluate metrics
y_pred = model.predict(X_test)
eval_metric(y_test, y_pred)
```

```python
# Preprocessing test data
test_data = df_test.drop("id", axis=1)
test_data = scaler.transform(test_data)
```

```python
# Predict on test data
y_pred = model.predict(test_data).flatten()
```

```python
# Create DataFrame for submission
output = pd.DataFrame({'id': df_test.id, 'Rings': y_pred})
```

```python
# Save predictions to CSV
output.to_csv('Abalone_ANN_v6.csv', index=False)

print("Completed")
``` 
