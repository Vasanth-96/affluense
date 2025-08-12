from pydantic import BaseModel
from typing import List, Optional
from utils.enums import Sentiment


class ScrapedRequest(BaseModel):
    name: str
    company: str

class ScrapedCompanyDetail(BaseModel):
    name: str
    average_sentiment: Sentiment
    negative_news_flag: bool
    total_articles: int


class ScrapedResponse(BaseModel):
    result: Optional[List[ScrapedCompanyDetail]] = None