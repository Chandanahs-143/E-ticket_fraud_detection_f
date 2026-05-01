# import pandas as pd
# import pickle
# from sklearn.ensemble import RandomForestClassifier

# # Load dataset
# df = pd.read_csv("eticket_fraud_data.csv")

# # Drop unnecessary columns (hour excluded as per original training)
# df = df.drop(['transaction_id', 'user_id', 'hour'], axis=1)

# # One-hot encoding for categorical features
# df = pd.get_dummies(df, columns=['device_type', 'location'])

# # Split features and label
# X = df.drop('is_fraud', axis=1)
# y = df['is_fraud']

# # Train Random Forest
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X, y)

# # Save model and column structure
# pickle.dump(model, open("model.pkl", "wb"))
# pickle.dump(X.columns.tolist(), open("columns.pkl", "wb"))

# print("✅ Model training completed!")
# print(f"   Features: {X.columns.tolist()}")
# print(f"   Training accuracy: {model.score(X, y):.4f}")




import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("eticket_fraud_data.csv")

print("Columns:", df.columns)

# Drop unnecessary columns safely
df = df.drop(columns=['transaction_id', 'user_id'], errors='ignore')

# Fix column names (IMPORTANT)
df.rename(columns={
    'num_tickets': 'tickets',
    'ticket_count': 'tickets'
}, inplace=True)

# Check again
print("After rename:", df.columns)

# One-hot encoding
df = pd.get_dummies(df, columns=['device_type', 'location'])

# Split
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("columns.pkl", "wb"))

print("✅ Training completed!")
print("Features used:", X.columns.tolist())
