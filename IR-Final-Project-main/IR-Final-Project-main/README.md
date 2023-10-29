# Social Media Information Response for Natural Disaster

<img src="https://github.com/ayush-shinde/IR-Final-Project/blob/main/img/website_img.png?raw=true" width="1800">

### Introduction
The increasing prevalence of natural disasters worldwide has highlighted the importance of effective disaster response strategies. Social media platforms, such as Twitter, have emerged as crucial channels for communication and information exchange during such events. This project seeks to explore the potential of leveraging machine learning models and natural language processing techniques to analyze social media data for improved disaster response and management.

### About Dataset
The dataset used in this project is the Natural Hazards Twitter Dataset which contains tweets related to various natural disasters. The dataset is publicly available on GitHub and comprises tweets that were collected during significant natural disaster events, such as hurricanes, earthquakes, and wildfires. The Dataset contains not only the text of the tweets but also metadata such as timestamp, geolocation, and user profile information. This additional information can be beneficial for understanding the context and impact of the disaster, as well as for identifying the most affected areas or urgent needs.

### Data Preprocessing

* Handling missing data: We filled in or removed missing data points to avoid inconsistencies during the analysis.
* Feature extraction: We identified and extracted relevant features from the metadata, such as geolocation and timestamp, for possible integration into the machine learning models. 

To ensure accurate analysis and model implementation, we carried out data preprocessing and cleaning tasks. The following data cleaning techniques were applied to the dataset:

1. Removing irrelevant content: We removed tweets that were not related to natural disasters.
2. Text normalization: We converted all text to lowercase and removed special characters, punctuation marks, and numbers.
3. Removing URLs and mentions: We removed any URLs and mentions present in the tweets.
4. Tokenization: We tokenized the text, splitting it into individual words.
5. Stopword removal: We removed common stopwords that don't provide significant value for sentiment analysis.
6. Lemmatization: We lemmatized the words, reducing them to their base forms.


### Analysis
#### Models Used
We used the following machine learning models in this project to examine and classify the sentiment of tweets associated with natural disasters:

* BERT
* DeBERTa
* RoBERTa
* K-Nearest Neighbors (KNN)
* Decision Tree
* Multiple Neural Network (MNN)
* Support Vector Machine (SVM)
* FLAN-UL2

![RoBERTa]

<img src="https://github.com/ayush-shinde/IR-Final-Project/blob/main/img/roberta.png?raw=true" width="1000">

#### FLAN-UL2 Implementation
The FLAN-UL2 (Flexible Language Adapter with Universal Language 2) model is a recently developed model with promising potential in natural language understanding tasks. We implemented this model alongside the others to evaluate its effectiveness in sentiment classification. The implementation involved fine-tuning the model on the preprocessed dataset and testing its performance against the other models.

#### Performance Evaluation
We assessed the performance and accuracy of each model in classifying tweet sentiment. The findings emphasize the advantages and drawbacks of each model concerning prediction accuracy, computational time, and other relevant metrics. The analysis offers valuable insights into which models are most appropriate for this specific classification task.

#### Results
Our analysis results indicate that some models, such as BERT and RoBERTa, achieved high accuracy in sentiment classification, while others, like KNN and Decision Tree, had lower accuracy. Overall, the study showcases the potential of using sophisticated natural language processing techniques and machine learning models to enhance disaster response initiatives through social media analysis.

![Model Performances](https://github.com/ayush-shinde/IR-Final-Project/blob/main/img/cs547-final%20project-12.png?raw=true)

#### Conclusion
This project demonstrates that social media data, particularly from Twitter, can be an invaluable asset for understanding user sentiment during natural disasters and guiding disaster response initiatives. By analyzing this data using various machine learning models, we can gain insights into which models excel in sentiment classification and make more informed decisions regarding disaster relief approaches.


*About the python files*
<br>
Required Libaries
1. transformers
2. sentence_transformers
3. tqdm
4. datasets
5. evaluate
6. peft

Required changes in code:
1. Change the directory path to dataset folder
