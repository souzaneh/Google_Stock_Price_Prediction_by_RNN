# 📈 Google Stock Price Prediction using RNN Models

This project focuses on **predicting the future stock prices of Google (GOOGL)** using historical daily price data and comparing the performance of multiple deep learning models.

---

## 🎯 Project Objective

The objective of this project is to forecast the stock price of Google based on past price data. The predictions are performed using four different deep learning models:

- **SimpleRNN**
- **LSTM**
- **GRU**
- **Hybrid LSTM-GRU**

The final output includes graphical visualization of the predictions on the test data and a quantitative evaluation of each model's performance.

---

## ⚙️ Technologies and Tools

- **Language:** Python
- **Libraries:** 
  - `TensorFlow`, `Keras` (for model building and training)
  - `NumPy`, `Pandas` (for data manipulation)
  - `Matplotlib` (for visualization)
  - `Scikit-learn` (for data preprocessing and scaling)
  - `Keras Tuner` (for hyperparameter optimization)
  - `yfinance` (for data collection)
- **Execution Environment:** Google Colab, Jupyter Notebook, VS Code

---

## 📅 Dataset

- **Source:** Collected using the `yfinance` library.
- **Asset:** Google stock (Ticker: GOOGL).
- **Date Range:** From **January 1, 2010** to **October 31, 2024**.
- **Frequency:** Daily closing prices.

---

## 🔄 Data Preparation

1. Downloaded historical daily stock prices using `yfinance`.
2. Used only the **Close** prices, normalized to values between **0 and 1**.
3. The sequence generation function (`sequence_data`) prepared the data such that:
   - The **last 10 days' prices** are used to predict the stock prices for the **next 2 days**.
4. The data was split into:
   - **80% Training set**
   - **20% Test set**

---

## 🏗️ Model Architecture & Training

- All four models(SimpleRNN,LSTM,GRU,Hybrid LSTM-GRU) were implemented with **two layers** and approximately **50 neurons**.
- Each model was trained for **100 epochs**.
- For the **SimpleRNN** and **LSTM** models, hyperparameter tuning was performed using **Keras Tuner** to optimize the number of layers and neurons.
- All models used:
  - **Adam optimizer**
  - **Mean Absolute Error (MAE)** as the loss function.

---

## 📏 Evaluation Metrics

- Model performances were evaluated using:
  - **Validation Loss**
  - **Visual comparison** of actual vs. predicted stock prices.
- The **Hybrid LSTM-GRU model** showed the lowest validation loss and delivered more accurate predictions, as reflected in both the metrics and plotted graphs.

---

## 📊 Visualizations

- The project includes graphical comparisons of:
  - Actual vs. Predicted prices for each model.
  - Validation loss curves for performance comparison.

> *(Graphs will be added soon.)*

---

## 🔍 Limitations and Future Work

- Limited computational resources restricted the depth of hyperparameter tuning and the number of training epochs.
- Increasing the number of epochs and applying more advanced hyperparameter optimization could further improve prediction accuracy.

---

## 🚀 How to Run

1. Open the `stock_prediction_google.ipynb` notebook in:
   - **Google Colab**
   - **Jupyter Notebook**
   - **VS Code** with Jupyter extension
2. Install required libraries (if needed) using:
pip install -r requirements.txt


3. Run the notebook cells to download the data, train the models, and visualize the predictions.

---

## 👩‍💻 Author

**Souzaneh Sehati**  
GitHub: [souzaneh](https://github.com/souzaneh)

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only** and does not constitute financial or investment advice.


