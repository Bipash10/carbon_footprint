from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pickle
import numpy as np
import hashlib  
from fastapi.middleware.cors import CORSMiddleware

# FastAPI app initialization
app = FastAPI()

# Middleware for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


users_db = {}

# Model and encoders loading
model = joblib.load("stacking_model.pkl")
scaler = pickle.load(open("scaler.pkl", "rb"))
one_hot_encoders = {
    "housing": joblib.load("encoding_logic_housing.pkl"),
    "water": joblib.load("encoding_logic_water.pkl"),
    "trash": joblib.load("encoding_logic_trash.pkl"),
    "heating": joblib.load("encoding_logic_heating.pkl"),
}
label_encoders = {
    "size": joblib.load("size_mapping.pkl"),
    "electricity": joblib.load("electricity_mapping.pkl"),
    "plastic_waste": joblib.load("plastic_waste_mapping.pkl"),
    "dairy": joblib.load("dairy_mapping.pkl"),
    "poultry": joblib.load("poultry_mapping.pkl"),
    "mutton": joblib.load("mutton_mapping.pkl"),
    "pork": joblib.load("pork_mapping.pkl"),
    "fish": joblib.load("fish_mapping.pkl"),
    "packaged": joblib.load("packaged_food_amount_mapping.pkl"),
    "diet_percentage": joblib.load("diet_percentage_mapping.pkl"),
    "travel_distance": joblib.load("travel_distance_mapping.pkl"),
    "fuel_economy": joblib.load("fuel_economy_mapping.pkl"),
    "flight_hours": joblib.load("flight_hours_mapping.pkl"),
}

# Input data model for prediction
class InputData(BaseModel):
    housing: str
    water: str
    trash: str
    heating: str
    size: str
    electricity: str
    plastic_waste: str
    dairy: str
    poultry: str
    mutton: str
    pork: str
    fish: str
    packaged: str
    diet_percentage: str
    travel_distance: str
    fuel_economy: str
    flight_hours: str

# User data model for login/registration
class User(BaseModel):
    email: str
    password: str


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(stored_password: str, provided_password: str) -> bool:
    return stored_password == hash_password(provided_password)


@app.post("/register")
async def register_user(user: User):
    if user.email in users_db:
        raise HTTPException(status_code=400, detail="User already exists")

    # Hash the password before storing
    hashed_password = hash_password(user.password)
    users_db[user.email] = hashed_password
    return {"message": "User registered successfully"}


@app.post("/login")
async def login_user(user: User):
    if user.email not in users_db:
        raise HTTPException(status_code=400, detail="User not found")
    
    
    stored_password = users_db[user.email]
    if not verify_password(stored_password, user.password):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    return {"message": "Login successful"}

# Prediction endpoint 
@app.post("/predict")
def predict(data: InputData):
    try:
        
        processed_data = preprocess_data(data)

        
        prediction = model.predict(processed_data.reshape(1, -1))[0]

        
        return {"carbon_footprint": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def health_check():
    return {"status": "API is running"}


def preprocess_data(data: InputData):
    # Initialization
    one_hot_encoded_values = []
    label_encoded_values = []


    for feature, encoder in one_hot_encoders.items():
        try:
            encoded = encoder.transform([[getattr(data, feature)]])[0]
            one_hot_encoded_values.extend(encoded)
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid value for {feature}: {getattr(data, feature)}"
            )

    
    for feature, encoder in label_encoders.items():
        try:
            encoded_value = encoder[getattr(data, feature)]
            label_encoded_values.append(encoded_value)
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid value for {feature}: {getattr(data, feature)}"
            )

    label_encoded_values = np.array(label_encoded_values).reshape(1, -1)
    standardized_values = scaler.transform(label_encoded_values)

   
    processed_data = np.hstack([one_hot_encoded_values, standardized_values[0]])

    # Check for feature mismatch
    expected_feature_count = model.n_features_in_
    if processed_data.shape[0] != expected_feature_count:
        raise HTTPException(
            status_code=400,
            detail=f"Feature mismatch. Expected {expected_feature_count} features, but got {processed_data.shape[0]}."
        )

    return processed_data
