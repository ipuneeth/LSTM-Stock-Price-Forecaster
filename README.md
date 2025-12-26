# 📈 LSTM Stock Price Forecaster

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=flat-square&logo=keras)
![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)

## 📖 Project Overview

This project implements a **Deep Learning model** to forecast daily stock closing prices. By utilizing **Long Short-Term Memory (LSTM)** networks, the model is capable of learning long-term dependencies in time-series data, making it superior to traditional linear regression for capturing market volatility.

The goal was to build a pipeline that preprocesses raw financial data, trains a neural network, and evaluates its predictive power against unseen future data.

---

## 🚀 Key Features

* **Data Pipeline:** Fetches raw CSV data and processes it using a **Sliding Window** technique (Look-back period: 60 days).
* **Robust Preprocessing:** Implements `MinMaxScaler` to normalize data for optimal LSTM convergence.
* **Deep Learning Architecture:** A stacked LSTM model with **Dropout regularization** to prevent overfitting.
* **Evaluation Metrics:** Validated using **RMSE** (Root Mean Squared Error) and **MAE** (Mean Absolute Error).
* **Visualization:** Comprehensive plots for Training Loss and Actual vs. Predicted prices.

---

## 🛠️ Technologies Used

* **Core:** Python 3.10+
* **Modeling:** TensorFlow, Keras
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Matplotlib

---

## 📊 Methodology

1.  **Data Acquisition:** Loaded historical stock data (Tesla/Synthetic).
2.  **Preprocessing:**
    * Converted the 'Close' price column to a numpy array.
    * Scaled values to the range `(0, 1)`.
    * Created sequential datasets where `X` is the past 60 days and `y` is the 61st day.
3.  **Model Training:**
    * **Layer 1:** LSTM (50 units, return sequences).
    * **Layer 2:** LSTM (50 units).
    * **Dense Layers:** 25 neurons -> 1 output neuron.
    * **Optimizer:** Adam.
4.  **Prediction:** Inverse transformed the output to get actual price values for comparison.

---

## 📉 Results

The model was tested on the last 20% of the dataset (unseen data).

* **Root Mean Squared Error (RMSE):** 17.48
* **Mean Absolute Error (MAE):** 13.16

> **Visual Performance:**
> The model successfully captures the general trend of the stock price. While it smooths out some of the extreme daily noise, it reacts effectively to major market shifts.

---

## 💻 How to Run

This project is optimized for **Google Colab**.

1.  Clone this repository.
2.  Open the notebook file (`LSTM_Stock_Prediction.ipynb`) in Google Colab or Jupyter Lab.
3.  Install dependencies (if running locally):
    ```bash
    pip install numpy pandas matplotlib tensorflow scikit-learn
    ```
4.  Run all cells sequentially.

---

## 👤 Author

**A Puneeth Chowdhary**
* **Connect:** https://github.com/ipuneeth

---
