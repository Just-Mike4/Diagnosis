from flask import Flask, request, jsonify, render_template 
import numpy as np
import pandas as pd
import pickle
import keras
from keras.models import load_model
app = Flask(__name__)

model= load_model("diagmodel.h5")
@app.route("/")
def home():
          return render_template("index.html")


@app.route("/predict",methods=["POST"])
def predict():
          value=request.form.to_dict()
          values=list(value.values())
          print(values)
          values=np.array(values)
          values=values.reshape(1,-1)
          v=[]
          for value in values:
                    value=value.astype('float')
                    v.append(value)
          v=np.array(v)
          print(v)
          prediction=model.predict(v)
          pred=np.argmax(prediction,axis=1)

          if pred == 0:
                    pred='Diarrhea'
          elif pred == 1:
                    pred='Malaria'
          else:
                    pred=='Mensturation'

          return render_template("index.html",prediction_text = "The diagnosis is {}".format(pred))

if __name__=="__main__":
          app.run(debug=True)