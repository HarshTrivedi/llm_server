from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def generate():
    return {"hello world"}
