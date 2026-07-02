# ⚡Energy Load Forecaster

An intelligent, time-series forecasting web application built to predict hourly electrical load consumption. Powered by a mathematically optimized **SARIMAX** machine learning model, this tool accounts for historical seasonal trends and real-time exogenous variables (temperature) to deliver accurate, lightning-fast predictions.

---

## 📸 Preview & Screenshots


![Application Dashboard](https://via.placeholder.com/800x450.png?text=UI+Dashboard+Preview)
> *The main dashboard where users input start/end timestamps and forecasted temperatures.*

![Prediction Results](https://via.placeholder.com/800x450.png?text=Prediction+Results+Table)
> *The model dynamically outputs unscaled, real-world energy load predictions within a clean, animated UI.*

---

## ✨ Key Features

* **Advanced Time-Series Forecasting:** Utilizes a SARIMAX `(4,1,0)x(1,1,0,24)` architecture to capture complex daily (24-hour) seasonalities.
* **Exogenous Variable Support:** Dynamically adjusts load predictions based on temperature inputs.
* **Low-Memory Optimization:** The training pipeline (`train.py`) is explicitly engineered to bypass RAM allocation limits using approximate diffuse initialization and memory-safe covariance tracking.
* **Instant Web Serving:** The web server (`app.py`) loads pre-compiled model weights directly from disk, ensuring zero lag and zero training overhead during HTTP requests.
* **Custom UI Design:** Built with pure modular CSS and keyframe animations—completely avoiding heavy utility frameworks like Tailwind for a lightweight, bespoke frontend.

---

## 🛠️ Tech Stack

**Backend & Machine Learning:**
* **Python 3.12+**
* **Flask** (Web Framework)
* **Statsmodels** (SARIMAX Time-Series Engine)
* **Scikit-Learn** (MinMaxScaler for data distribution normalization)
* **Pandas & NumPy** (Data wrangling and timeline structuring)

**Frontend:**
* **HTML5**
* **Modular CSS** (Custom animations, table styling, and responsive layout)

---

## 📂 Project Structure

```text
├── app.py                 # The Flask web server and prediction routing
├── train.py               # The heavy-lifting ML pipeline (Trains model & scaler)
├── energy.csv             # Historical dataset (Required for training)
├── templates/
│   └── index.html         # Frontend user interface
├── model.pkl              # Generated post-training (Fully converged weights)
└── scaler.pkl             # Generated post-training (MinMax scaling bounds)

```

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/yourusername/energy-load-forecaster.git](https://github.com/yourusername/energy-load-forecaster.git)
cd energy-load-forecaster

```

### 2. Install Dependencies

Ensure you have Python installed, then run:

```bash
pip install flask pandas numpy statsmodels scikit-learn

```

### 3. Add Your Dataset

Place your historical `energy.csv` dataset into the root directory. Ensure it contains at least a `timestamp` column, a `load` column, and a `temp` (or `temperature`) column.

### 4. Train the Model (Required First Step)

To avoid server memory crashes, the model must be trained offline first. Run the training script to generate `model.pkl` and `scaler.pkl`:

```bash
python train.py

```

*(Note: This step may take a few minutes as the model optimizes its mathematical weights.)*

### 5. Launch the Web Server

Once the training script says `SUCCESS`, you can start the Flask application:

```bash
python app.py

```

The server will start instantly. Open your browser and navigate to `http://127.0.0.1:5001`.

---

## 💡 Usage Guide

1. Enter a **Start Timestamp** (e.g., `2014-12-30 01:00:00`).
2. Enter an **End Timestamp** (e.g., `2014-12-30 05:00:00`).
3. Enter the anticipated **Temperature** for that window (e.g., `27.0`).
4. Click Predict. The backend will map the timeline, scale the inputs, query the pre-trained SARIMAX model, unscale the results, and render a formatted table instantly.

---

*Designed & Developed for efficient, real-world energy consumption forecasting.*
