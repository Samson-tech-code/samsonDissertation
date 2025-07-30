# samsonDissertation

This repository contains a simple COVID-19 dashboard built with Flask and Pandas. It loads the Johns Hopkins time series dataset and exposes an API for querying confirmed cases by date range and country. A basic web UI displays the data using Chart.js.

In addition to the dashboard, the repo includes a small mental health dataset and an analysis pipeline demonstrating descriptive statistics, bivariate tests, regression models and basic machine learning classifiers.

## Setup

```bash
pip install -r requirements.txt
```

## Running the server

```bash
python dashboard/app.py
```

Open `http://localhost:5000` in your browser and use the form to select country and date range. A line chart will display the confirmed case counts.

## Running the analysis pipeline

```bash
python analysis/pipeline.py
```

The script will output basic statistics to the console and save a `ghq12_distribution.png` plot to the `analysis/` folder.

## Tests

```bash
pytest -q
```
