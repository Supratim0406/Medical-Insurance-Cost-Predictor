from fastapi import FastAPI, Request # Fast API - Creates the web API application # Request - Required to pass request object to HTML templates
from fastapi.templating import Jinja2Templates # Renders HTML pages (frontend)
from pydantic import BaseModel  # Validates API input using Pydantic
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware # Allows frontend & backend to communicate
from fastapi import HTTPException


# Load saved model, and scaler
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Initialize FastAPI app
app = FastAPI(title="Medical Insurance Cost Predictor")

# Tells FastAPI My HTML files are inside the templates/ folder
templates = Jinja2Templates(directory="templates")

# Enable CORS - To call backend APIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input Schema (Request Validation) -> 1) Defines expected input JSON 2) Automatically validates data types
class InsuranceInput(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

# Encoding Categorical Variables
sex_map = {"male": 1, "female": 0}
smoker_map = {"yes": 1, "no": 0}
region_map = {
    "southeast": 0,
    "southwest": 1,
    "northwest": 2,
    "northeast": 3,
}

# Home route
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction route
# 1.Accepts POST request at /predict 2.Automatically parses JSON into InsuranceInput 3.Validates input
@app.post("/predict")
def predict_charges(input_data: InsuranceInput):

    # Normalize input
    sex = input_data.sex.lower().strip()
    smoker = input_data.smoker.lower().strip()
    region = input_data.region.lower().strip()

    # Convert Input to DataFrame
    new_data = pd.DataFrame([{
    "age": input_data.age,
    "sex": sex_map[sex],
    "bmi": input_data.bmi,
    "children": input_data.children,
    "smoker": smoker_map[smoker],
    "region": region_map[region]
}])


    # Scale only numeric columns
    new_data[["age", "bmi", "children"]] = scaler.transform(
        new_data[["age", "bmi", "children"]]
    )

    # Ensure correct column order
    new_data = new_data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
    
    # Prediction will return numpy array
    prediction = model.predict(new_data)[0]
    return {"predicted_charges": round(float(prediction), 2)}
