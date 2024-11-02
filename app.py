import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template
from waitress import serve



app = Flask(__name__)

# Laster inn modellen vha. pickle.
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
        print(f"Model loaded successfully: {type(model)}")
        
        # Access the best model
        best_model = model.best_estimator_
        print(f"Best model type: {type(best_model)}")
except Exception as e:
    print(f"Error loading the model: {str(e)}")




# Henter medianverdier til hver numeriske kolonne. Disse har jeg regnet ut i analysebiten.
median_values = {
    'alder': 65.294985,
    'utdanning': 12.0,
    'blodtrykk': 77.0,
    'hvite_blodlegemer': 10.1992188,
    'hjertefrekvens': 100.0,
    'respirasjonsfrekvens': 24.0,
    'kroppstemperatur': 36.69531,
    'lungefunksjon': 226.65625,
    'serumalbumin': 3.5,
    'kreatinin': 1.19995117,
    'natrium': 137.0,
    'blod_ph': 7.429688,
    'glukose': 133.0,
    'blodurea_nitrogen': 24.0,
    'urinmengde': 2502.0,
    'antall_komorbiditeter': 2.0,
    'koma_score': 0.0,
    'adl_stedfortreder': 1.0,
    'fysiologisk_score': 24.34765625,
    'overlevelsesestimat_2mnd': 0.6954345705,
    'overlevelsesestimat_6mnd': 0.551879883
}


# Definerer binære alternativer på nettsiden.
binary_features = {
    'kjønn': ['female', 'male'],
    'inntekt': ['$11-$25k', '$25-$50k', '>$50k', 'under $11k', 'nan'],
    'etnisitet': ['asian', 'black', 'hispanic', 'other', 'white', 'nan'],
    'sykdomskategori': ['ARF/MOSF', 'COPD/CHF/Cirrhosis', 'Cancer', 'Coma'],
    'sykdom_underkategori': ['ARF/MOSF w/Sepsis', 'CHF', 'COPD', 'Cirrhosis', 'Colon Cancer', 'Coma', 'Lung Cancer', 'MOSF w/Malig'],
    'kreft': ['metastatic', 'no', 'yes'],
    'diabetes': ['yes', 'no'],
    'demens': ['yes', 'no'],
    'dnr_dag': ['yes', 'no'],
    'dnr_status': ['dnr før innleggelse', 'dnr ved innleggelse', 'None']
}

@app.route('/')
def home():
    return render_template('index.html', median_values=median_values, binary_features=binary_features)
  

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.form.to_dict()

        processed_input = {
            'alder': float(input_data.get('alder', 0)),
            'utdanning': float(input_data.get('utdanning', 0)),
            'blodtrykk': float(input_data.get('blodtrykk', 0)),
            'hvite_blodlegemer': float(input_data.get('hvite_blodlegemer', 0)),
            'hjertefrekvens': float(input_data.get('hjertefrekvens', 0)),
            'respirasjonsfrekvens': float(input_data.get('respirasjonsfrekvens', 0)),
            'kroppstemperatur': float(input_data.get('kroppstemperatur', 0)),
            'lungefunksjon': float(input_data.get('lungefunksjon', 0)),
            'serumalbumin': float(input_data.get('serumalbumin', 0)),
            'kreatinin': float(input_data.get('kreatinin', 0)),
            'natrium': float(input_data.get('natrium', 0)),
            'blod_ph': float(input_data.get('blod_ph', 0)),
            'glukose': float(input_data.get('glukose', 0)),
            'blodurea_nitrogen': float(input_data.get('blodurea_nitrogen', 0)),
            'urinmengde': float(input_data.get('urinmengde', 0)),
            'antall_komorbiditeter': float(input_data.get('antall_komorbiditeter', 0)),
            'koma_score': float(input_data.get('koma_score', 0)),
            'adl_stedfortreder': float(input_data.get('adl_stedfortreder', 0)),
            'fysiologisk_score': float(input_data.get('fysiologisk_score', 0)),
            'overlevelsesestimat_2mnd': float(input_data.get('overlevelsesestimat_2mnd', 0)),
            'overlevelsesestimat_6mnd': float(input_data.get('overlevelsesestimat_6mnd', 0)),
            'diabetes': 1.0 if input_data.get('diabetes') == 'yes' else 0.0,
            'demens': 1.0 if input_data.get('demens') == 'yes' else 0.0,
            'dnr_dag': 1.0 if input_data.get('dnr_dag') == 'yes' else 0.0,
            'kjønn_female': 1.0 if input_data.get('kjønn') == 'female' else 0.0,
            'kjønn_male': 1.0 if input_data.get('kjønn') == 'male' else 0.0,
            'inntekt_$11-$25k': 1.0 if input_data.get('inntekt') == '$11-$25k' else 0.0,
            'inntekt_$25-$50k': 1.0 if input_data.get('inntekt') == '$25-$50k' else 0.0,
            'inntekt_>$50k': 1.0 if input_data.get('inntekt') == '>$50k' else 0.0,
            'inntekt_under $11k': 1.0 if input_data.get('inntekt') == 'under $11k' else 0.0,
            'inntekt_nan': 1.0 if input_data.get('inntekt') is None else 0.0,
            'etnisitet_asian': 1.0 if input_data.get('etnisitet') == 'asian' else 0.0,
            'etnisitet_black': 1.0 if input_data.get('etnisitet') == 'black' else 0.0,
            'etnisitet_hispanic': 1.0 if input_data.get('etnisitet') == 'hispanic' else 0.0,
            'etnisitet_other': 1.0 if input_data.get('etnisitet') == 'other' else 0.0,
            'etnisitet_white': 1.0 if input_data.get('etnisitet') == 'white' else 0.0,
            'etnisitet_nan': 1.0 if input_data.get('etnisitet') is None else 0.0,
            'sykdomskategori_ARF/MOSF': 1.0 if input_data.get('sykdomskategori') == 'ARF/MOSF' else 0.0,
            'sykdomskategori_COPD/CHF/Cirrhosis': 1.0 if input_data.get('sykdomskategori') == 'COPD/CHF/Cirrhosis' else 0.0,
            'sykdomskategori_Cancer': 1.0 if input_data.get('sykdomskategori') == 'Cancer' else 0.0,
            'sykdomskategori_Coma': 1.0 if input_data.get('sykdomskategori') == 'Coma' else 0.0,
            'sykdom_underkategori_ARF/MOSF w/Sepsis': 1.0 if input_data.get('sykdom_underkategori') == 'ARF/MOSF w/Sepsis' else 0.0,
            'sykdom_underkategori_CHF': 1.0 if input_data.get('sykdom_underkategori') == 'CHF' else 0.0,
            'sykdom_underkategori_COPD': 1.0 if input_data.get('sykdom_underkategori') == 'COPD' else 0.0,
            'sykdom_underkategori_Cirrhosis': 1.0 if input_data.get('sykdom_underkategori') == 'Cirrhosis' else 0.0,
            'sykdom_underkategori_Colon Cancer': 1.0 if input_data.get('sykdom_underkategori') == 'Colon Cancer' else 0.0,
            'sykdom_underkategori_Coma': 1.0 if input_data.get('sykdom_underkategori') == 'Coma' else 0.0,
            'sykdom_underkategori_Lung Cancer': 1.0 if input_data.get('sykdom_underkategori') == 'Lung Cancer' else 0.0,
            'sykdom_underkategori_MOSF w/Malig': 1.0 if input_data.get('sykdom_underkategori') == 'MOSF w/Malig' else 0.0,
            'kreft_metastatic': 1.0 if input_data.get('kreft') == 'metastatic' else 0.0,
            'kreft_no': 1.0 if input_data.get('kreft') == 'no' else 0.0,
            'kreft_yes': 1.0 if input_data.get('kreft') == 'yes' else 0.0,
            'dnr_status_dnr før innleggelse': 1.0 if input_data.get('dnr_status') == 'dnr før innleggelse' else 0.0,
            'dnr_status_dnr ved innleggelse': 1.0 if input_data.get('dnr_status') == 'dnr ved innleggelse' else 0.0,
            'dnr_status_None': 1.0 if input_data.get('dnr_status') == 'None' else 0.0,
        }
        
        features_df = pd.DataFrame([processed_input])

        #Predikerer oppholdslengde med den gitte modellen.
        prediction = model.predict(features_df)
        prediction_result = prediction[0]

        # Sjekker at den predikerte veriden ikke er negativ. 
        if prediction_result < 0:
            prediction_result = 0

       # Returnerer resultatet til nettsiden.
        return render_template('./index.html',
                               prediction_text=f"Predikert oppholdslengde: {round(prediction_result, 2)} dager",
                               median_values=median_values,
                               binary_features=binary_features)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('./index.html',
                               prediction_text="An error occurred during prediction.",
                               median_values=median_values,
                               binary_features=binary_features)



















if __name__ == '__main__':
    app.run(port=8080, debug=True)