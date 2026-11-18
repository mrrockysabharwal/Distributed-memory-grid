"""
FastAPI application entrypoint for Distributed Memory Grid.
Maintained by Member 4 (Full-stack, Integration & DevOps Lead).
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router

app = FastAPI(
    title="Distributed Memory Grid (DMG) API Gateway",
    description="Real-time distributed LLM inference engine control plane and streaming gateway.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/health")
def healthcheck():
    return {"status": "ok", "service": "DMG Gateway"}


if __name__ == "__main__":
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)

