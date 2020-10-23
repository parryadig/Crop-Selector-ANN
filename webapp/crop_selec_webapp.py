import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from flask import Flask, request, jsonify, render_template

# load model
def load():
	global model
	model = load_model("crop_predictor.h5")
	global graph
	graph = tf.get_default_graph()

app = Flask(__name__)

df = pd.read_csv("crop_data.csv")
y = pd.get_dummies(df["label"], prefix="")
crop_arr = y.columns.values

@app.route("/")
def running():
    return render_template("./index.html")

@app.route("/predict", methods=["POST"])
def predicting():
    data1 = request.form["a"]
    data2 = request.form["b"]
    data3 = request.form["c"]
    data4 = request.form["d"]
    arr = np.array([[data1, data2, data3, data4]])
    load()
    with graph.as_default():
        pred = crop_arr[np.argmax(model.predict(arr[0:1]))]
    return render_template("after.html", data=pred)





