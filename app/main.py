import lightgbm
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def predict():
    # load the model
    model = lightgbm.Booster(model_file="path/to/saved/model.txt")

    # use the model to make a prediction
    prediction = model.predict(data)

    return {"prediction": prediction}
