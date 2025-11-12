from fastapi import FastAPI, HTTPException
import uvicorn
from fastapi.responses import Response

# Создаём приложение FastAPI
app = FastAPI()


@app.get("/status/")
async def get_status():
    status = {"status": "running"}
    return status

@app.get("/shot")
def get_image():
    with open("static.png", "rb") as f:
        image_data = f.read()
    return Response(content=image_data, media_type="image/png")


# Запуск сервера с явным указанием хоста и порта
uvicorn.run(app, host="127.0.0.1", port=8001)
