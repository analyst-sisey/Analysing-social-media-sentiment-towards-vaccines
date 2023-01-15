import gradio as gr
# Creating a gradio app using the inferene API
App = gr.Interface.load("huggingface/allevelly/sentiment_analysis_of_tweets_on_covid",
  title="sentiment analysis of tweets on covid19 Vaccines", description ="sentiment analysis of tweets on covid19 Vaccines using DistilBERT model",
 allow_flagging=False, examples=[["Type your messgage about covid vaccines above"]]
)

App.launch()
