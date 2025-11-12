import numpy as np
from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__, template_folder='templates')

# Load model with absolute path
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = pickle.load(open(model_path, "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # get values from form
            sepal_length = float(request.form["Sepal_Length"])
            sepal_width = float(request.form["Sepal_Width"])
            petal_length = float(request.form["Petal_Length"])
            petal_width = float(request.form["Petal_Width"])
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            output = model.predict(features)[0]
            return render_template("index.html", prediction_text=f"The Flower Name is {output}")
        except Exception as e:
            return render_template("index.html", prediction_text=f"Error: {e}")
    else:
        # GET request
        return render_template("index.html")
