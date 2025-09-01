from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

app = Flask(__name__)

# Load and preprocess data
def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
        required_columns = ['budget', 'revenue', 'genre', 'actors_y', 'directors_y']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in the dataset")

        df['budget'] = pd.to_numeric(df['budget'].replace(r'[\$,]', '', regex=True), errors='coerce')
        df['revenue'] = pd.to_numeric(df['revenue'].replace(r'[\$,]', '', regex=True), errors='coerce')

        df['success'] = pd.cut(df['revenue'],
                               bins=[0, 1e6, 1e7, 5e7, 1e8, np.inf],
                               labels=['flop', 'average', 'hit', 'superhit', 'blockbuster'])

        df = df.dropna()

        le = {}
        for col in ['genre', 'actors_y', 'directors_y']:
            le[col] = LabelEncoder()
            df[col] = le[col].fit_transform(df[col].astype(str).str.strip().str.lower())

        scaler = StandardScaler()
        df['budget'] = scaler.fit_transform(df[['budget']])

        return df, le, scaler

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None, None, None

# Function to train the model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Encode input values
def encode_input(input_value, encoder, column_name):
    try:
        normalized_value = input_value.strip().lower()
        return encoder[column_name].transform([normalized_value])[0]
    except ValueError:
        print(f"Warning: '{input_value}' is a new {column_name} not seen during training.")
        return -1

# Load and preprocess the data at startup
df, le, scaler = load_and_preprocess_data('/Users/ammardadani/Desktop/Sem 7 projects/bda project/data/finaldatasetmovies (1).csv')
X = df[['budget', 'genre', 'actors_y', 'directors_y']]
y = df['success']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using the training data
model = train_model(X_train, y_train)

# Evaluate the model on the test data
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        movie_name = request.form['movie_name']  # New: Movie Name
        budget = float(request.form['budget'])
        genre = request.form['genre']
        actors = request.form['actors']
        director = request.form['director']

        # Scale the budget and encode other inputs
        budget_scaled = scaler.transform(pd.DataFrame([[budget]], columns=['budget']))[0][0]
        genre_encoded = encode_input(genre, le, 'genre')
        actors_encoded = encode_input(actors, le, 'actors_y')
        director_encoded = encode_input(director, le, 'directors_y')

        # Prepare the input for prediction
        input_data = pd.DataFrame([[budget_scaled, genre_encoded, actors_encoded, director_encoded]],
                                  columns=['budget', 'genre', 'actors_y', 'directors_y'])

        # Predict the movie's success
        success = model.predict(input_data)[0]

        # Render the result template and pass the movie name and prediction
        return render_template('result.html', prediction=success, movie_name=movie_name)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
