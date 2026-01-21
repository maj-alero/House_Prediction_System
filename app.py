from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Correct path: Ensure this matches your GitHub file structure
MODEL_PATH = 'model/house_price_model.pkl'

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    print(f"CRITICAL ERROR: {MODEL_PATH} not found!")

@app.route('/')
def index():
    neighborhoods = sorted(['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 
                            'Somerst', 'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 
                            'NridgHt', 'NAmes', 'SawyerW', 'IDOTRR', 'MeadowV'])
    return render_template('index.html', neighborhoods=neighborhoods)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            # Capture inputs
            data = {
                'OverallQual': [int(request.form.get('OverallQual', 5))],
                'GrLivArea': [float(request.form.get('GrLivArea', 0))],
                'TotalBsmtSF': [float(request.form.get('TotalBsmtSF', 0))],
                'GarageCars': [int(request.form.get('GarageCars', 0))],
                'Neighborhood': [request.form.get('Neighborhood', 'NAmes')]
            }
            
            query_df = pd.DataFrame(data)
            prediction = model.predict(query_df)[0]
            
            formatted_price = f"${prediction:,.2f}"
            
            # Re-list neighborhoods so the dropdown doesn't disappear on results page
            neighborhoods = sorted(['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 
                                    'Somerst', 'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 
                                    'NridgHt', 'NAmes', 'SawyerW', 'IDOTRR', 'MeadowV'])
            
            return render_template('index.html', 
                                   prediction_text=formatted_price,
                                   neighborhoods=neighborhoods)
    except Exception as e:
        return f"Prediction Error: {str(e)}"

if __name__ == "__main__":
    # Render requires binding to 0.0.0.0 and the PORT env variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)