from flask import Flask, jsonify, request, render_template
import pandas as pd
from datetime import datetime

app = Flask(__name__)

DATA_PATH = 'data/time_series_covid19_confirmed_global.csv'

# Load the global confirmed cases dataset
covid_df = pd.read_csv(DATA_PATH)

# Melt the dataset so dates are rows
melted = covid_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='date', value_name='cases')
# Convert date column to datetime for filtering
melted['date'] = pd.to_datetime(melted['date'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def api_data():
    """Return filtered COVID-19 data as JSON."""
    start = request.args.get('start')
    end = request.args.get('end')
    country = request.args.get('country')

    df = melted
    if country:
        df = df[df['Country/Region'] == country]
    if start:
        df = df[df['date'] >= pd.to_datetime(start)]
    if end:
        df = df[df['date'] <= pd.to_datetime(end)]

    grouped = df.groupby('date')['cases'].sum().reset_index()
    grouped['date'] = grouped['date'].dt.strftime('%Y-%m-%d')
    return jsonify(grouped.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
