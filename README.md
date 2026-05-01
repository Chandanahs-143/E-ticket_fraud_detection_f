# 🛡️ E-Ticket Fraud Detection System

## Setup & Run

1. Install dependencies:
   pip install -r requirements.txt

2. Train the model (generates model.pkl + columns.pkl):
   python train.py

3. Run the Streamlit app:
   streamlit run app.py

4. Login with: admin / admin123

## Project Structure

| File | Description |
|------|-------------|
| app.py | Main Streamlit web application |
| train.py | Model training script |
| eticket_fraud_data.csv | Dataset (1000 transactions) |
| model.pkl | Trained Random Forest model |
| columns.pkl | Feature column names for inference |
| requirements.txt | Python dependencies |

## ML Model
- **Algorithm**: Random Forest Classifier (100 estimators)
- **Prediction**: Uses predict_proba() to output fraud probability (0–100%)
- **Features**: amount, tickets_booked, device_type (OHE), location (OHE)
- **Note**: 'hour' was excluded during training and inference

## Risk Levels
- 🔴 HIGH: ≥65% fraud probability
- 🟡 MEDIUM: 35–64%
- 🟢 LOW: <35%
