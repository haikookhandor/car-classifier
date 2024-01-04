## Technical Know-How

## 1. Text Extraction
- **PDF Text Extraction:** Utilized the PyPDF2 library to programmatically extract raw text from PDF documents. Extracted text content from each page of the PDF, preserving the document's structure and formatting.


## 2. Model Selection
- **Multinomial Naive Bayes Classifier:** Chose the Multinomial Naive Bayes algorithm due to its efficiency in handling NLP tasks. Car Brochure PDF's contain a lot of words where many are redundant or irrelevant. Occurrence of one feature does not affect the probability of occurrence of the other feature. For small sample sizes, Naïve Bayes can outperform the most powerful alternatives. Being relatively robust, easy to implement, fast, and accurate, this was the reason for the model selection.

## 4. Text Vectorization
- **TF-IDF Vectorization:** Utilized TfidfVectorizer to convert raw text into a numerical format, considering term frequency and inverse document frequency to highlight the importance of specific terms in distinguishing documents.

## 5. Model Deployment
- **Joblib Model Saving:** Saved the trained model using the joblib library, serializing the model to disk for later deployment. 

## 6. Streamlit App
- **Streamlit:** Developed a user-friendly web application using Streamlit, integrating it seamlessly with the backend model for user interaction.

