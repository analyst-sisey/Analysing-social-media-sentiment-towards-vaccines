import gradio as gr
from gradio.mix import Parallel, Series
app1 = gr.Interface.load("huggingface/allevelly/sentiment_analysis_of_tweets_on_covid")
app2 =gr.Interface.load("huggingface/allevelly/Analysing_socialMedia_sentiment_on_vaccines")
#app3= gr.Interface(my_language_model,"text","text")
Parallel(app1,app2).launch()