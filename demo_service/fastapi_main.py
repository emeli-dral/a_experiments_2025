from fastapi import FastAPI, Request,Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from classifier import Classifier

app = FastAPI()
templates = Jinja2Templates(directory="templates")
classifier = Classifier()

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("simple_page.html", {
        "request": request,
        "text": "",
        "prediction_message": ""
    })

@app.post("/", response_class=HTMLResponse, name="index_page")
def predict(request: Request, text: str = Form(...)):
    prediction = classifier.get_result_message(text)
    return templates.TemplateResponse("simple_page.html", {
        "request": request,
        "text": text,
        "prediction_message": prediction
    })

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)