# main.py

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import PyPDF2
import os
import joblib

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def read_data_from_directory(directory_path):
    data = []
    labels = []
    for label in os.listdir(directory_path):
        label_path = os.path.join(directory_path, label)
        if os.path.isdir(label_path):
            for file_name in os.listdir(label_path):
                print(file_name)
                file_path = os.path.join(label_path, file_name)
                if file_path.endswith('.pdf'):
                    text = extract_text_from_pdf(file_path)
                    data.append(text)
                    labels.append(label)
    return data, labels


all_directory = 'All'
X_raw, y = read_data_from_directory(all_directory)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
X_train_processed = vectorizer.fit_transform(X_train_raw)
X_test_processed = vectorizer.transform(X_test_raw)
classifier = MultinomialNB()
classifier.fit(X_train_processed, y_train)
# Predict on the testing data
y_pred = classifier.predict(X_test_processed)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2%}')

# Display classification report
print('\nClassification Report:\n', classification_report(y_test, y_pred))

# Save the trained model using joblib
model_filename = 'model.joblib'
joblib.dump(classifier, model_filename)
print(f'Trained model saved to {model_filename}')
joblib.dump(X_test_processed, 'X_test_processed.joblib')
joblib.dump(y_test, 'y_test.joblib')
