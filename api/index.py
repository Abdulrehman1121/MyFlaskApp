import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # get values from form
            sepal_length = float(request.form["Sepal_Length"])
            sepal_width  = float(request.form["Sepal_Width"])
            petal_length = float(request.form["Petal_Length"])
            petal_width  = float(request.form["Petal_Width"])

            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            output = model.predict(features)[0]

            return render_template("index.html", prediction_text=f"The Flower Name is {output}")
        except Exception as e:
            return render_template("index.html", prediction_text=f"Error: {e}")
    else:
        # GET request
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

#Link
#IP address
#Domain Name
#Frontend
#Backend
#Bug
#Request
#Hosting
#Server
#Database
#HTML
#Post
#Get
#https://flower-predictor-3022.onrender.com/