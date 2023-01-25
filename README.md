# Analysing-social-media-sentiment-towards-vaccines using hugging Face NLP models

* This project uses the NLP technique of sentiment analysis in analysing social media sentiment towards vaccines.
Natural Language Processing is a field of artificial intelligence that deals with the interaction between computers and humans using natural language. 
   * Sentiment analysis is a specific application of NLP that involves identifying and extracting subjective information from text, such as determining the emotional tone of a statement.  It uses machine learning and lexical analysis to classify text as positive, negative, or neutral in sentiment. It is often used in social media monitoring, customer service, and market research.
    * The importance of NLP and sentiment analysis lies in the ability to process and understand large amounts of unstructured text, image and voice data, which is the most common format in which information is available. Sentiment analysis are used in various applications such as marketing, customer service, and political campaign.
    * The objective of this project is to develop a NLP machine learning model to assess if a Twitter post related to vaccinations is positive, neutral, or negative.
    This project builds 2 sentiment analysis models and compare their performance. These models were built using pre-trained models from huggingface.

    * Huggingface is an AI-powered open-source platform for natural language processing (NLP) that provides access to pre-trained models, datasets, and evaluation metrics. It allows developers to easily train, test and deploy natural language processing models and build applications such as chatbots, language translation, and text summarization. 


# How to run the project

* Create the Python's virtual environment that isolates the required libraries of the project;
Activate the Python's virtual environment so that the Python kernel & libraries will be those of the isolated environment;
Install the required libraries/packages listed in the requirements.txt file;

# To view and run the project code
* open the Sentiment_Analysis_of_tweets_on_covid.ipynb file. This file was created using colab notebook

# To run the sentiment analysis Gradio app 
* run the command "python GradioApp.py" at your python prompt

# To run compare the result of the 2 models in a single Gradio app 
* run the command "python SentimentApp.py"

# Please find below the link to the models on HuggingFace Hub
* The DistilBERT-base-uncased Model - https://huggingface.co/allevelly/sentiment_analysis_of_tweets_on_covid
* The roBERTa-base model - https://huggingface.co/allevelly/Analysing_socialMedia_sentiment_on_vaccines 

# Please find below the link to the Gradio Apps on HuggingFace Spaces
* App for sentiment analysis - https://huggingface.co/spaces/allevelly/sentimentApp
* App that compares the 2 models - https://huggingface.co/spaces/allevelly/sentimentModelComparison
