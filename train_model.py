import pandas as pd

# Load the CSV file
data = pd.read_csv('your_dataset.csv')

# Display first 5 rows
print(data.head())

# Check columns
print(data.columns)
# Select features
df = data[['mag', 'depth', 'latitude', 'longitude']]

# Drop missing values
df = df.dropna()

# Create a new column: High Risk (1 if mag >= 5, else 0)
df['high_risk'] = df['mag'].apply(lambda x: 1 if x >= 5 else 0)

# Check data
print(df.head())
from sklearn.model_selection import train_test_split

# Features (input variables)
X = df[['depth', 'latitude', 'longitude']]

# Target (high risk or not)
y = df['high_risk']

# Split data: 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.ensemble import RandomForestClassifier

# Initialize model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))
import seaborn as sns
import matplotlib.pyplot as plt

# Plot Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Scatter plot of earthquakes by depth & magnitude
plt.scatter(df['depth'], df['mag'], alpha=0.5)
plt.xlabel('Depth')
plt.ylabel('Magnitude')
plt.title('Earthquake Magnitude vs Depth')
plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
data = pd.read_csv('your_dataset.csv')  # replace with your CSV filename

# Add a simple prediction label: Severe (1) if mag >= 5, else 0
data['Prediction'] = data['mag'].apply(lambda x: 1 if x >= 5.0 else 0)

# Features and labels
X = data[['latitude', 'longitude', 'depth', 'mag']]
y = data['Prediction']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'model.pkl')

print("✅ Model trained and saved as 'model.pkl'")



