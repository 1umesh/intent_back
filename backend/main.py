from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import uvicorn
from utils.re_ranker import reranker

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")

# Load model
model = joblib.load("/Users/umeshmeena/python_projects/lead_intent/conversion_pipeline.pkl")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")#, response_class=HTMLResponse
async def predict(
    request: Request,
    Age: int = Form(...),
    Gender: str = Form(...),
    LeadSource: str = Form(...),
    TimeSpent: int = Form(...),
    PagesViewed: int = Form(...),
    EmailSent: int = Form(...),
    DeviceType: str = Form(...),
    FormSubmissions: int = Form(...),
    Downloads: int = Form(...),
    CTR_ProductPage: float = Form(...),
    ResponseTime: int = Form(...),
    FollowUpEmails: int = Form(...),
    SocialMediaEngagement: int = Form(...),
    PaymentHistory: str = Form(...),
    query: str = Form(...)
):

    # Prepare data
    user_data = pd.DataFrame([{
        'Age': Age,
        'Gender': Gender,
        'LeadSource': LeadSource,
        'TimeSpent (minutes)': TimeSpent,
        'PagesViewed': PagesViewed,
        'EmailSent': EmailSent,
        'DeviceType': DeviceType,
        'FormSubmissions': FormSubmissions,
        'Downloads': Downloads,
        'CTR_ProductPage': CTR_ProductPage,
        'ResponseTime (hours)': ResponseTime,
        'FollowUpEmails': FollowUpEmails,
        'SocialMediaEngagement': SocialMediaEngagement,
        'PaymentHistory': PaymentHistory
    }])
    print("🧪 Incoming user_data:")
    print(user_data)
    print("🧪 Columns:", user_data.columns)

    prediction = model.predict(user_data)[0]
    proba = model.predict_proba(user_data)[0][1]
    result = reranker(proba, query)
    return result

# Run with: uvicorn main:app --reload