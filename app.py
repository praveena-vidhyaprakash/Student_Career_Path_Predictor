from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
interest_encoder = joblib.load("interest_encoder.pkl")
career_encoder = joblib.load("career_encoder.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        interest = request.form["interest"]
        math = int(request.form["math"])
        science = int(request.form["science"])
        commerce = int(request.form["commerce"])
        arts = int(request.form["arts"])

        interest_encoded = interest_encoder.transform([interest])[0]

        input_df = pd.DataFrame([{
            "SubjectInterest": interest_encoded,
            "MathScore": math,
            "ScienceScore": science,
            "CommerceScore": commerce,
            "ArtsScore": arts
        }])

        pred = model.predict(input_df)[0]
        career = career_encoder.inverse_transform([pred])[0]
        prediction = f"Recommended Career Path: {career}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
