"""
FastAPI application entry point
"""
print("server may take time to start")

from fastapi import FastAPI
from utils.async_logger import logger
from models.models import ScrapedRequest
from process import scrape_and_analyze_news

# Initialize FastAPI app
app = FastAPI()
logger.info("server loaded")


@app.get("/", summary="Root Endpoint")
async def root():
    """Root endpoint to check if API is running"""
    return {"message": "Affluense API is running!"}


@app.post("/scrape-and-flag", summary="Scrape data on demand")
async def scrape_and_flag(request: ScrapedRequest):
    """
    Main endpoint to scrape and analyze news data for companies
    """
    return await scrape_and_analyze_news(request)


@app.get("/health", summary="Health Check Endpoint")
async def health_check():
    """Health check endpoint to verify API status"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
