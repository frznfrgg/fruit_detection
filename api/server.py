from fastapi import FastAPI
from fastapi.responses import JSONResponse
import base64
from io import BytesIO
from PIL import Image
import uvicorn
import os

app = FastAPI()

@app.get("/status/")
async def get_status():
    status = {"status": "running"}
    return status

@app.get("/get_last")
def get_images_base64():
    image_paths = ["static.png", "static.png"]
    images_b64 = []

    for path in image_paths:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        images_b64.append({
            "data": f"data:image/png;base64,{encoded}"
        })

    response_data = {
        "ready": True,
        "images": images_b64
    }

    return JSONResponse(content=response_data)

uvicorn.run(app, host="127.0.0.1", port=8001)
