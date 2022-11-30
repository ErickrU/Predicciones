# create an API using FASTAPI
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from tempfile import NamedTemporaryFile
import pickle
import joblib
import pandas as pd
import numpy as np

# create an instance of the API
app = FastAPI()

# allow CORS
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello from prediction service API"}

# define the route
@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):

    file_content = await file.read()

    temp_file = NamedTemporaryFile(delete=False)

    temp_file.write(file_content)

    bike_data = pd.read_csv(temp_file.name)

    y = bike_data['rentals'].values

    y = y.reshape(-1,1)

    x = bike_data[['season','mnth', 'holiday','weekday','workingday','weathersit','temp', 'atemp', 'hum', 'windspeed']].values

    model = pickle.load(open("finalized_mode.sav",'rb'))

    predictions = model.predict(x)

    print(predictions)

    temp_file.close()

    return JSONResponse(content=jsonable_encoder({"predictions": predictions.tolist()}))

@app.post("/error")
async def error_api(file: UploadFile = File(...)):
    file_content = await file.read()

    temp_file = NamedTemporaryFile(delete=False)

    temp_file.write(file_content)

    bike_data = pd.read_csv(temp_file.name)

    y = bike_data['rentals'].values

    y = y.reshape(-1,1)

    x = bike_data[['season','mnth', 'holiday','weekday','workingday','weathersit','temp', 'atemp', 'hum', 'windspeed']].values

    model = pickle.load(open("finalized_mode.sav",'rb'))

    predictions = model.predict(x)

    temp_file.close()

    mse = mean_squared_error(y, predictions)

    r2 = r2_score(y, predictions)

    rmse = np.sqrt(mse)

    return JSONResponse(content=jsonable_encoder({"mse": mse,"rmse": rmse ,"r2": r2}))


