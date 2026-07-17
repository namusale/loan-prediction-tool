"""
The module works to receive input data from the frontend,checks data and passes the
data to the model for prediction. The output of the prediction is sent back to the
frontend for display.
"""
import feature_performance
import utils
import feature_engineering
import feature_analysis
import feature_selection
from model import data_preprocess


from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json

app = FastAPI()

origins = ["http://localhost:5174",
            "http://127.0.0.1:5174",
            "http://localhost:5173",
            "http://127.0.0.1:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic check for format complaince
class Data(BaseModel):
    cibil_score: float = 0.0
    loan_term:float = 0.0
    income_annum: float = 0.0
    education_not_graduate: str = 'Yes'
    loan_amount: float = 0.0
    self_employed: str = 'No'
    luxury_assets_value: float = 0.0


# Testing fastAPI get
@app.get("/")
async def learn():
    return {"message": "We are reminding "}


# fastapi post for receiving from front end and passes it for prediction in model.py

@app.post("/predict/")
def receive_data(data: Data):
    received_data = data.model_dump()
    outcome = list(data_preprocess(received_data))
    print(outcome[0])
    return {"message": "We have received the request",
            "look": data.model_dump(),
            "output": json.dumps(int(outcome[0]))
            }

if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    # utils.utilities()
    # feature_engineering.feature_eng()
    # feature_analysis.feature_analyse()
    # feature_performance.feature_performance()
    # feature_selection.aggregate()
    # model.model_run(data)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
