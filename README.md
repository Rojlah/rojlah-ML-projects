Diabetes Data Prediction Using Machine Learning

Overview
This project focuses on predicting diabetes using machine learning techniques in Python. The model is trained on the PIMA Diabetes dataset, containing various health parameters, to classify individuals as diabetic or non-diabetic.

Features
- Data preprocessing and feature selection
- Implementation of Support Vector Machine (SVM) with a linear kernel
- Standardization using `StandardScaler`
- Model evaluation using accuracy score
- Predictive system for user input data

 Dataset
The dataset used in this project includes health-related features such as:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome (0 - Non-Diabetic, 1 - Diabetic)

Technologies Used
- Python
- Pandas & NumPy (Data Processing)
- Scikit-learn (Machine Learning Models)
- Matplotlib & Seaborn (Data Visualization)
- Google Colab

 Installation
To run this project on Google Colab, follow these steps:

1. Open Google Colab and upload the notebook:
   - Go to [Google Colab](https://colab.research.google.com/)
   - Click on "File" > "Upload Notebook" and select the `.ipynb` file

2. Install necessary dependencies by running:
   ```python
   !pip install -r requirements.txt
   ```
   (If a `requirements.txt` file is not available, install individual packages as needed.)

3. Upload the dataset to Colab:
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```

 Usage
 1. Import Dependencies
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
```

2. Load and Explore Dataset
```python
diabetes_dataset = pd.read_csv('/content/diabetes.csv')
print(diabetes_dataset.head())
print(diabetes_dataset.shape)
print(diabetes_dataset.describe())
print(diabetes_dataset['Outcome'].value_counts())
```

3. Data Preprocessing
```python
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
```

4. Split Data into Training and Testing Sets
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
```

5. Train the Model
```python
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
```

6. Evaluate Model Performance
```python
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data: ', training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data: ', test_data_accuracy)
```

7. Making a Predictive System
```python
input_data = (4, 110, 92, 0, 0, 37.6, 0.191, 30)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)
prediction = classifier.predict(std_data)

if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is Diabetic')
```
Results
- Accuracy score of training data: ~78.66%
- Accuracy score of test data: ~77.27%
- Predictive system successfully classifies a given input instance

Future Enhancements
- Implementing Deep Learning models for better accuracy
- Deploying the model as a web application
- Adding more features for improved prediction

Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.


Contact
For any inquiries, reach out to: [rojlahrajkumar@gmail.com]

