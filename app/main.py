import lightgbm as lgb
import os
from fastapi import FastAPI

from pydantic import BaseModel, constr, conint
import logging
import pandas as pd
import numpy as np

my_logger = logging.getLogger()
my_logger.setLevel(logging.DEBUG)


MODEL_DIR_PATH = (
    os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir) + "/models"
)
MODEL_NAME = "model_ckpt_1690.txt"
TH_PREDICTIONS = 0.5

model_path = os.path.join(MODEL_DIR_PATH, MODEL_NAME)

delay_predictor = lgb.Booster(model_file=model_path)

app = FastAPI()


class Data(BaseModel):
    """Base model for input data validation."""

    origin: constr(to_upper=True, max_length=4)
    destination: constr(to_upper=True, max_length=4)
    distance_trip: int
    mails_data: int
    cargo_data: int
    number_checked_luggages: int
    flights_per_day: int
    previous_is_delayed_same_day: bool
    day_of_week: conint(gt=0,lt = 31)
    day_of_year: conint(gt=0, lt = 366)
    departure_minutes: conint(gt = 0, lt = 60)
    departure_hour: conint(gt = 0, lt = 24)
    month: conint(gt=0, lt = 12)
    total_number_passengers: int
    flights_within_hour: int


@app.post("/predict")
async def predict(data: Data):
    """Function to generate predictions"""

    my_logger.debug("Loading data")
    inputs = pd.DataFrame([data.dict()])

    inputs.origin = inputs.origin.astype("category")
    inputs.destination = inputs.destination.astype("category")
    inputs.previous_is_delayed_same_day = inputs.previous_is_delayed_same_day.astype("category")

    my_logger.debug("prediction")
    predictions = delay_predictor.predict(inputs)

    my_logger.debug("Mapping predictions")
    predictions = np.where(predictions >= TH_PREDICTIONS, "delayed", "not delayed")
    return {"prediction": list(predictions)}
