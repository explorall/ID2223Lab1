import gradio as gr
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("wine_model", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model.pkl")
print("Model downloaded")

def wine(Type, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide,
         total_sulfur_dioxide, density, ph, sulphates, alcohol):
    #Maps type to int with default white
    if Type=="red":
        Type=1
    else:
        Type=0


    print("Calling function") 
    df = pd.DataFrame([[Type, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide,
                        total_sulfur_dioxide, density, ph, sulphates, alcohol]], 
                      columns=["type", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
                        "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "ph", "sulphates", "alcohol"])
    print("Predicting")
    print(df)
    res = model.predict(df) 
    print(res)
          
    return "Predicted quality: "+str(res[0])
        
demo = gr.Interface(
    fn=wine,
    title="Wine Quality Predictive Analytics",
    description="Experiment with features to predict wine quality.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Textbox(default="white", label="Type"),
        gr.inputs.Number(default=0.34, label="Volatile acidity"),
        gr.inputs.Number(default=0.32, label="Citric acid"),
        gr.inputs.Number(default=5.4, label="Residual sugar"),
        gr.inputs.Number(default=0.056, label="Chlorides"),
        gr.inputs.Number(default=31.0, label="Free sulfur dioxide"),
        gr.inputs.Number(default=116.0, label="Total sulfur dioxide"),
        gr.inputs.Number(default=0.995, label="Density"),
        gr.inputs.Number(default=3.2, label="pH"),
        gr.inputs.Number(default=0.53, label="Sulphates"),
        gr.inputs.Number(default=10.5, label="Alcohol precentage"),
        ],
    outputs=gr.Label())

demo.launch(debug=True)