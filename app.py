from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import gzip
import traceback

# ==========================================
# 1. GLOBAL INITIALIZATION (LIGHTWEIGHT)
# ==========================================

print("Loading pre-trained SARIMAX model weights and scaler...")
try:
    with gzip.open("model.pkl", "rb") as f:
        model_results = pickle.load(f)

    with gzip.open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    print("Server structures successfully loaded! Ready for predictions.")
except Exception as e:
    print(f"CRITICAL ERROR during startup: Could not load model/scaler files. Ensure you have run train.py first. Details: {e}")

app = Flask(__name__)

# ==========================================
# 2. FLASK ROUTES & FORECASTING LOGIC
# ==========================================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict_load():
    def get_timestamp(field_name):
        timestamp = request.form.get(field_name)
        if not timestamp:
            return f"<h1 style='color:red'>Missing value for {field_name}</h1>"
        print(f"{field_name.capitalize()}: {timestamp}")
        return timestamp

    def get_temperature():
        try:
            temp = float(request.form.get('temp'))
            print("Temperature:", temp)
            return temp
        except (ValueError, TypeError):
            return f"<h1 style='color:red'>Invalid temperature value provided</h1>"

    def forecast_load(start_timestamp, end_timestamp, temp):
        try:
            # Generate the hourly datetime index for the user's requested window
            timestamp_index = pd.date_range(start=start_timestamp, end=end_timestamp, freq='h')
            num_predictions = timestamp_index.shape[0]
            
            # Prepare exogenous features matching the forecast duration
            exogenous_data = np.array([temp] * num_predictions).reshape(-1, 1)

            # Perform high-speed forecasting using the fully trained parameters
            predictions_scaled = model_results.predict(start=start_timestamp, end=end_timestamp, exog=exogenous_data)
            print("Predictions (scaled):", predictions_scaled.values)

            # Inverse transform the scaled predictions back to actual, readable units
            predictions_unscaled = scaler.inverse_transform(predictions_scaled.values.reshape(-1, 1)).flatten()
            print("Predictions (unscaled):", predictions_unscaled)

            # Create a DataFrame containing the unscaled load outputs
            results_df = pd.DataFrame(predictions_unscaled, index=timestamp_index, columns=['load'])
            results_df.index.name = 'timestamp'

            # Build the clean UI response string
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Energy Consumption</title>
                <style>
                    @keyframes borderAnimation {{
                        0% {{ border-color: #4CAF50; box-shadow: 0 0 25px #4CAF50; }}
                        25% {{ border-color: #FF5733; box-shadow: 0 0 25px #FF5733; }}
                        50% {{ border-color: #FFC300; box-shadow: 0 0 25px #FFC300; }}
                        75% {{ border-color: #3498DB; box-shadow: 0 0 25px #3498DB; }}
                        100% {{ border-color: #4CAF50; box-shadow: 0 0 25px #4CAF50; }}
                    }}
                    @keyframes borderAnimation2 {{
                        0%, 100% {{ color: orangered; }}
                        50% {{ color: orange; }}
                    }}
                    body {{ background-color: #1a1a1a; font-family: 'Montserrat', sans-serif; }}
                    h1 {{
                        text-align: center;
                        letter-spacing: 5px;
                        animation: borderAnimation2 5s infinite;
                        margin-top: 30px;
                    }}
                    table {{
                        margin: 20px auto;
                        border-radius: 10px;
                        background-color: #f9f9f9;
                        animation: borderAnimation 5s infinite;
                        width: 50%;
                        text-align: center;
                        border-collapse: separate;
                        border-spacing: 0;
                        overflow: hidden;
                    }}
                    th, td {{ padding: 15px; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #4CAF50; color: white; }}
                    tr:hover {{ background-color: #f5f5f5; }}
                </style>
            </head>
            <body>
                <h1>Load Prediction Results</h1>
                <table>
                    <thead>
                        <tr>
                            <th style="border-top-left-radius:10px">Timestamp</th>
                            <th style="border-top-right-radius:10px">Load Output</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(f"<tr><td>{str(ts)}</td><td>{load:.2f}</td></tr>" for ts, load in zip(results_df.index, results_df['load']))}
                    </tbody>
                </table>
            </body>
            </html>
            """
            return html_content
        except Exception as e:
            return f"<h1 style='color:red'>Error during prediction: {e}</h1><pre>{traceback.format_exc()}</pre>"

    # Extract forms
    start_timestamp = get_timestamp('s_timestamp')
    end_timestamp = get_timestamp('e_timestamp')
    temp = get_temperature()

    # Validation early exits
    if isinstance(start_timestamp, str) and start_timestamp.startswith('<h1'): return start_timestamp
    if isinstance(end_timestamp, str) and end_timestamp.startswith('<h1'): return end_timestamp
    if isinstance(temp, str) and temp.startswith('<h1'): return temp

    return forecast_load(start_timestamp, end_timestamp, temp)

if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=True, port=5001)