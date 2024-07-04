from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import io
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import numpy as np
from pydantic import BaseModel, ValidationError
from httpx import AsyncClient
from openai import OpenAI
import os

app = FastAPI(
    title="API",
    description="L'API permet l'entraînement et la prédiction via l'aide d'un modèle.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"Hello": "World"}


# Chargement du fichier CSV
file_path = 'vgsalesGlobale.csv'
data = pd.read_csv(file_path)

# Afficher les premières lignes et les informations sur les données
data_info = data.info()
first_rows = data.head()

data_info, first_rows


@app.post("/train/")
async def train_model(file: UploadFile = File(...)):
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")
    
    # Lire le fichier CSV
    contents = await file.read()
    data = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    
    # Vérifier les colonnes nécessaires
    required_columns = ['Platform', 'Year', 'Genre', 'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
    if not all(column in data.columns for column in required_columns):
        raise HTTPException(status_code=400, detail=f"Missing one or more required columns: {required_columns}")
    
    # Séparation des caractéristiques (X) et de la cible (y)
    X = data.drop(columns=['Unnamed: 0', 'Genre', 'Name', 'Rank'])
    y = data['Genre']
    
    # Convertir y en format catégoriel
    y = to_categorical(y)
    
    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Définir le modèle TensorFlow
    model = Sequential()
    print(X_train, X_train.shape)
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    
    # Compiler le modèle
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Entraîner le modèle
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.2)
    
    # Sauvegarde du modèle
    model.save('model.h5')
    
    # Prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)
    y_test_classes = y_test.argmax(axis=1)
    
    # Évaluation du modèle
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    classification_report_str = classification_report(y_test_classes, y_pred_classes)
    
    return JSONResponse(content={
        "accuracy": accuracy,
        "classification_report": classification_report_str
    })

# Modèle Pydantic pour la structure de la requête de prédiction
class PredictionRequest(BaseModel):
    Platform: int
    Year: float
    Publisher: int
    NA_Sales: float
    EU_Sales: float
    JP_Sales: float
    Other_Sales: float
    Global_Sales: float

# Endpoint pour faire des prédictions
@app.post("/predict/")
async def predict(request: PredictionRequest):
    # Vérifier si le modèle existe
    if not os.path.exists('model.h5'):
        raise HTTPException(status_code=500, detail="Model not found. Please train the model first.")
    
    try:
        # Charger le modèle sauvegardé
        model = load_model('model.h5')
        
        # Convertir les données de la requête en format nécessaire pour le modèle
        data = np.array([[request.Platform, request.Year, request.Publisher, request.NA_Sales, request.EU_Sales, request.JP_Sales, request.Other_Sales, request.Global_Sales]])
        
        # Faire la prédiction
        prediction = model.predict(data)
        predicted_class = prediction.argmax(axis=1)[0]
        
        return JSONResponse(content={"predicted_genre": int(predicted_class)})
    
    except ValidationError as e:
        raise HTTPException(status_code=400, detail="Invalid input data")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Récupération de la clé API OpenAI
OPENAI_API_KEY = os.getenv("your_api_key")
OPENAI_API_URL = "https://api.openai.com/v1/engines/davinci-codex/completions"

# Modèle Pydantic pour la requête de l'utilisateur
class YearRequest(BaseModel):
    year: int

@app.get("/game_by_year/")
async def get_game_by_year(year: int):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key is not configured.")
    
    prompt = f"Give me a popular video game released in the year {request.year}."
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "max_tokens": 50
    }
    
    async with AsyncClient() as client:
        response = await client.post(OPENAI_API_URL, json=data, headers=headers)
    
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve data from OpenAI: {response.text}")
    
    response_data = response.json()
    game_info = response_data.get("choices")[0].get("text", "").strip()
    
    return {"game": game_info}

@app.get("/games/{year}")
async def get_games_by_year(year: int):
    client = OpenAI(
        api_key= "your_api_key"
    )

    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "Tu es un expert en jeux vidéo"},
        {"role": "user", "content": "tu vas m'aider à trouver des jeux en fonction de leur date de sortie !"},
        {"role": "assistant", "content": "Salut tout le monde !"},
        {"role": "user", "content": "trouve un jeu qui est sorti en : " + str(year)}
      ]
    )

    return response.choices[0].message.content

@app.get("/test_api_key/")
def test_api_key():
    return {"API Key": OPENAI_API_KEY}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)