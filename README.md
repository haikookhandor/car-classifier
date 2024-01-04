## Car Classification from PDF Brochures
The classifier is trained on text extracted from pdf files of car brochures. It learns the key features for each car type and predicts the car type for a test pdf. 

### Source
Dataset: https://www.auto-brochures.com/

Reason: It is the "Largest Collection of US Car Brochures on the Internet!" as claimed by the website. There are car brochures of over 80 brands for almost all models and that too over several years. This allows to incorporate diversity and a temporal component as car brochures change slightly over the years allowing the model to appreciate this difference and make better predictions. 

### How to use
1. The dataset (linked below) consists of 100 PDF files with a 4:1 Train:Test split ratio. Download it using the link given below.

2. The codebase already contains the trained model, test dataset in compressed form so dataset requirement can be omitted if you are having storage issues. If not, run main.py after setting it in appropriate path.

3. After changing the model parameters, dataset, etc., your new model is saved as a .joblib file.

4. Use this saved model and load it to test on any pdf file you wish to get inference on. 

5. DUe to storage issues, you need to simply change the files (model, test data) and see the resulting accuracy on the streamlit app.

### Dataset Drive Link:
https://drive.google.com/drive/folders/1va_6_-lLyKYdNeahxengfRy2u_dJFfzE?usp=sharing

### EDA:
Exploratory Data Analysis has been performed on the dataset incorporating some text-based experiments to understand the broad data structure of the documents. 

### Live-Deployment Link:
Please use this link to view the deployed code - https://huggingface.co/spaces/haikookhandor/car_classifier 
New link - https://huggingface.co/spaces/haikookhandor/trial

Note: Currently this shows only the performance on the test data, if you wish to see performance on custom dataset, follow instructions given in "How to use" section above. 

### Challenges Faced:
1. Dataset Curation: Since the dataset was to be chosen by us, there was a lot of thought and process that went behind this. First of all, choosing an appropriate site was necessary keeping in mind the type of cars and text-readable pdf files. Old car brochures have scanned images which makes them useless for our purpose.

2. Storage Issue: This was faced when I tried to load the dataset in it's entirety (~800 MB). I realized that saving the model and subsequently the test dataset would be necessary and the only way.

### Future Use:
1. Live upload and inference: Currently it is a very limited app where you need to replace the files in order to see real time inference. In the future, I want to replace this by an upload button feature where uploading the pdf would give the model prediction making it slightly better in usability.

2. Expand the dataset: The dataset is limited and might not be representative in car features and types. Expanding the dataset to other countries would help take into account the regional diversity and car feature advertised. One could also do a zero shot transfer classification where the model is trained on one region and tested on other. 


