from flask import Flask, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model and feature names
with open("cart_model.pkl", "rb") as f:
    model, feature_names = pickle.load(f)

@app.route('/')
def index():
    with open("index.html", "r", encoding="utf-8") as file:
        return file.read()

@app.route('/predict', methods=["POST"])
def predict():
    # Extract input features from form
    input_data = {
        'team1_score': float(request.form['team1_score']),
        'team2_score': float(request.form['team2_score']),
        'overs_left': float(request.form['overs_left']),
        'wickets_left': float(request.form['wickets_left']),
        'run_rate': float(request.form['run_rate']),
        # Add more if needed
    }

    # Create DataFrame and align with training features
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Predict
    prediction = model.predict(input_df)[0]

    # Return prediction
    return f"""
    <html>
    <head><title>Prediction Result</title>
    <link rel="stylesheet" href="styles.css"></head>
    <body>
        <div class="container">
            <h1>🏏 IPL Match Winner Prediction</h1>
            <h2 class="result">Predicted Match Winner: <span style='color:green'>{prediction}</span></h2>
            <br><a href="/">🔙 Predict Another</a>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(debug=True)
