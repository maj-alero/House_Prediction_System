from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# 1. Load the saved trained model
model = joblib.load('model/house_price_model.pkl')

@app.route('/')
def index():
    # List of neighborhoods for the dropdown (matches dataset categories)
    neighborhoods = sorted(['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 
                            'Somerst', 'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 
                            'NridgHt', 'NAmes', 'SawyerW', 'IDOTRR', 'MeadowV'])
    return render_template('index.html', neighborhoods=neighborhoods)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 2. Capture user inputs from the form
        data = {
            'OverallQual': [int(request.form['OverallQual'])],
            'GrLivArea': [float(request.form['GrLivArea'])],
            'TotalBsmtSF': [float(request.form['TotalBsmtSF'])],
            'GarageCars': [int(request.form['GarageCars'])],
            'Neighborhood': [request.form['Neighborhood']]
        }
        
        # 3. Pass input data to the model
        query_df = pd.DataFrame(data)
        prediction = model.predict(query_df)[0]
        
        # 4. Display result
        formatted_price = f"${prediction:,.2f}"
        return render_template('index.html', 
                               prediction_text=formatted_price,
                               neighborhoods=sorted(['CollgCr', 'Veenker', 'Crawfor', 'NoRidge']))

if __name__ == "__main__":
    app.run(debug=True)