import asyncio
from aiohttp import ClientSession
from bs4 import BeautifulSoup
from typing import Dict, List, AsyncGenerator, Tuple
from utils.threadpool import THREAD_POOL


class AsyncWebScraper:
    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    async def _fetch_html(self, session: ClientSession, url: str) -> Tuple[str, str]:
        """Fetch HTML and return (html, url) tuple"""
        try:
            async with session.get(url, timeout=self.timeout) as resp:
                try:
                    resp.raise_for_status()
                    html = await resp.text()
                    return html, url
                except Exception as e:
                    raise Exception(f"Failed to read response: {str(e)}")
        except asyncio.TimeoutError:
            raise Exception(f"Request timed out after {self.timeout} seconds")
        except Exception as e:
            raise Exception(f"Failed to fetch URL {url}: {str(e)}")

    def _parse_html(self, html: str, url: str) -> Dict:
        """Parse HTML in thread pool"""
        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Remove scripts and styles
            try:
                for script in soup(["script", "style"]):
                    script.extract()
            except Exception as e:
                raise Exception(f"Failed to clean HTML: {str(e)}")

            try:
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip()
                          for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)

                return {
                    'url': url,
                    'title': soup.title.string if soup.title else 'No title',
                    'content': text,
                    'status': 'success',
                    'length': len(text)
                }
            except Exception as e:
                raise Exception(f"Failed to extract text: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to parse HTML from {url}: {str(e)}")

    async def _process_fetch_result(self, fetch_task, loop):
        """Process a single fetch result through the thread pool"""
        try:
            html, url = await fetch_task
            # Immediately pass to thread pool
            return await loop.run_in_executor(THREAD_POOL, self._parse_html, html, url)
        except Exception as e:
            # Extract URL from the task for error reporting
            url = "unknown"
            return {
                'url': url,
                'title': None,
                'content': None,
                'status': 'failed',
                'error': str(e)
            }

    async def scrape_urls_stream_pipeline(self, urls: List[str]) -> AsyncGenerator[Dict, None]:
        """
        Pipeline approach: Fetch -> Parse -> Yield
        Each step happens as soon as the previous completes
        """
        async with ClientSession(headers={'User-Agent': 'Mozilla/5.0'}) as session:
            loop = asyncio.get_running_loop()

            # Start all fetch operations
            fetch_tasks = [
                asyncio.create_task(self._fetch_html(session, url))
                for url in urls
            ]

            # Process fetch results as they complete
            processing_tasks = []

            for completed_fetch in asyncio.as_completed(fetch_tasks):
                # As soon as fetch completes, start parsing in thread pool
                processing_task = asyncio.create_task(
                    self._process_fetch_result(completed_fetch, loop)
                )
                processing_tasks.append(processing_task)

            # Yield parsing results as they complete
            for completed_processing in asyncio.as_completed(processing_tasks):
                result = await completed_processing
                yield result

    async def scrape_urls_stream_immediate(self, urls: List[str]) -> AsyncGenerator[Dict, None]:
        """
        Even more immediate approach: Start parsing as soon as each fetch completes
        """
        if not urls:
            return

        try:
            async with ClientSession(headers={'User-Agent': 'Mozilla/5.0'}) as session:
                loop = asyncio.get_running_loop()

                async def fetch_and_parse(url: str):
                    """Fetch URL and parse in one go"""
                    try:
                        if not url or not isinstance(url, str):
                            raise ValueError(f"Invalid URL: {url}")

                        html, fetched_url = await self._fetch_html(session, url)
                        # Parse in thread pool
                        return await loop.run_in_executor(
                            THREAD_POOL, self._parse_html, html, fetched_url
                        )
                    except Exception as e:
                        return {
                            'url': url,
                            'title': None,
                            'content': None,
                            'status': 'failed',
                            'error': str(e)
                        }

                # Create tasks for all URLs
                try:
                    tasks = [
                        asyncio.create_task(fetch_and_parse(url))
                        for url in urls
                    ]
                except Exception as e:
                    raise Exception(f"Failed to create tasks: {str(e)}")

                # Yield results as they complete
                try:
                    for completed_task in asyncio.as_completed(tasks):
                        try:
                            result = await completed_task
                            yield result
                        except Exception as e:
                            yield {
                                'url': 'unknown',
                                'title': None,
                                'content': None,
                                'status': 'failed',
                                'error': f"Task completion failed: {str(e)}"
                            }
                except Exception as e:
                    raise Exception(f"Error processing tasks: {str(e)}")

        except Exception as e:
            yield {
                'url': 'unknown',
                'title': None,
                'content': None,
                'status': 'failed',
                'error': f"Critical scraper error: {str(e)}"
            }


scraper = AsyncWebScraper()
