from flask import Flask, request, render_template
import joblib
import pandas as pd
from utils import preprocess_data

app = Flask(__name__)

# Modell laden
model = joblib.load('lead_scorer_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Eingabedaten aus dem Formular
    age = int(request.form['age'])
    website_visits = int(request.form['website_visits'])
    email_interactions = int(request.form['email_interactions'])
    purchased_before = int(request.form['purchased_before'])
    
    # Daten in ein DataFrame umwandeln und vorverarbeiten
    input_data = pd.DataFrame([[age, website_visits, email_interactions, purchased_before]],
                              columns=['age', 'website_visits', 'email_interactions', 'purchased_before'])
    
    processed_data = preprocess_data(input_data)
    
    # Vorhersage mit dem besten Modell
    lead_score = model.predict(processed_data)[0]
    
    return f"Der Lead-Score für diesen Lead ist: {lead_score}"

if __name__ == '__main__':
    app.run(debug=True)
