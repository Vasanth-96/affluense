from asyncddgs import aDDGS
from typing import List, Dict, Any
import logging
from aiohttp.client_exceptions import ClientError


class DuckDuckGoAsyncSearcher:
    """
    using the async version of the free search tool,
    this class provides methods to search for various content types.
    """

    def __init__(self, region: str = "us-en", safesearch: str = "moderate"):
        self.region = region
        self.safesearch = safesearch

    async def _fetch_results(self, method: str, query: str, max_results: int) -> List[Dict[str, Any]]:
        try:
            async with aDDGS() as client:
                try:
                    search_method = getattr(client, method)
                    results = await search_method(
                        keywords=query,
                        region=self.region,
                        safesearch=self.safesearch,
                        max_results=max_results
                    )
                    return results
                except AttributeError as e:
                    logging.error(
                        f"Invalid search method '{method}': {str(e)}")
                    return []
                except ClientError as e:
                    logging.error(
                        f"API request failed for query '{query}': {str(e)}")
                    return []
        except Exception as e:
            logging.error(
                f"Unexpected error during DuckDuckGo search: {str(e)}")
            return []

    async def search_text(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        return await self._fetch_results("text", query, max_results)

    async def search_news(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        return await self._fetch_results("news", query, max_results)


duck_duck_go_searcher = DuckDuckGoAsyncSearcher()
