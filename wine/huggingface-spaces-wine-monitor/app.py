import gradio as gr
from PIL import Image
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()

dataset_api.download("Resources/textfiles/latest_wine.txt")
dataset_api.download("Resources/textfiles/actual_wine.txt")
dataset_api.download("Resources/images/df_wine_recent.png")
dataset_api.download("Resources/images/wine_confusion_matrix.png")

with gr.Blocks() as demo:
    with gr.Row():
      with gr.Column():
          gr.Label("Today's Predicted Quality\n" + open("latest_wine.txt","r").readline())
      with gr.Column():          
          gr.Label("Today's Actual Quality\n" + open("actual_wine.txt","r").readline())      
    with gr.Row():
      with gr.Column():
          gr.Label("Recent Prediction History")
          input_img = gr.Image("df_wine_recent.png", elem_id="recent-predictions")
      with gr.Column():          
          gr.Label("Confusion Maxtrix with Historical Prediction Performance")
          input_img = gr.Image("wine_confusion_matrix.png", elem_id="confusion-matrix")        

demo.launch()
