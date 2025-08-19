from models.models import ScrapedRequest
from typing import List, Dict

# uncomment for more results
def generate_initial_search_queries(request: ScrapedRequest) -> List[str]:
    # Generate initial search queries based on the request data
    queries = [
        f"{request.name} {request.company}",
        f"Who is {request.name} and what companies has he worked for?"
    ]
    return queries

def generate_follow_up_queries(company: str) -> List[str]:
    news_queries = [
        f"Latest news about {company}",
        f"Fraud allegations against {company}",
        f"any recent developments regarding {company}",
        f"any negative news about {company}"
    ]
    return news_queries

def extract_urls_from_results(results: List[Dict]) -> List[str]:
    urls = []
    for res in results:
        url = res.get('href') or res.get('url')
        if url:
            urls.append(url)
    return urls
