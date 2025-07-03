from fastapi import FastAPI
from app.routes import router

app = FastAPI(title="RecommendIt")
app.include_router(router)