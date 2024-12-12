# Spam Email Classifier using NLP and Machine Learning

This project demonstrates a spam email classification system using Natural Language Processing (NLP) techniques and a Naive Bayes algorithm implemented with Python. The code uses popular libraries like Pandas, Numpy, and scikit-learn for data preprocessing, feature extraction, and model training.

## Features
- Preprocesses email text data for classification
- Uses Bag-of-Words or TF-IDF for feature extraction
- Implements the Naive Bayes algorithm for spam classification
- Achieves effective spam detection performance

## Dependencies
Ensure you have the following Python libraries installed:
- pandas
- numpy
- scikit-learn

Install dependencies using pip:
```bash
pip install pandas numpy scikit-learn
```

## Usage
1. Clone the repository:
```bash
git clone <repository_url>
cd <repository_folder>
```
2. Open the `spam_classifier.ipynb` file in Jupyter Notebook or any Python IDE supporting Jupyter Notebooks.
3. Ensure the dataset is properly loaded and named `emails.csv`.
4. Execute all the cells sequentially.

## Code Overview
```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the dataset
emails = pd.read_csv('emails.csv')

# Preprocessing
emails['text'] = emails['text'].str.lower()
X = CountVectorizer().fit_transform(emails['text'])
y = emails['spam']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

## Dataset
The dataset used for this project should contain:
- `text`: Email content
- `spam`: Label indicating whether the email is spam (1) or not (0)

Ensure the dataset is properly formatted and saved as `emails.csv` in the project directory.

## Results
The Naive Bayes classifier demonstrates effective performance in distinguishing spam emails with an accuracy of approximately **X%** (replace with actual results).

## Contributions
Feel free to submit issues or pull requests to improve the project.

## License
This project is licensed under the MIT License.
