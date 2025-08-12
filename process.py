"""
Business logic for news scraping and sentiment analysis
"""
import time
import asyncio
from utils.async_logger import logger
from utils.sync_logger import sync_logger
from models.models import ScrapedRequest, ScrapedResponse, ScrapedCompanyDetail
from utils.helper_functions import (
    generate_initial_search_queries,
    extract_urls_from_results,
    generate_follow_up_queries
)
from services.async_duck_duck_go import duck_duck_go_searcher
from services.async_threadpool_simple_scraper import scraper
from services.async_langchain import langchain_service
from utils.config import CLASSIFIER_TYPE
from utils.enums import ClassifierType, Sentiment
from classifiers.finbert import finbert_classifier
from classifiers.zeroshort import zeroshort_classifier
from utils.threadpool import THREAD_POOL



def classify_sync(content: str, title: str, company: str):
    """Synchronous classification function"""
    try:
        if CLASSIFIER_TYPE == ClassifierType.FINBERT:
            print(f"Using FinBERT Sentiment Classification for {company}")
            return finbert_classifier.process_news(content, title, company=company)
        elif CLASSIFIER_TYPE == ClassifierType.ZEROSHORT:
            print(f"Using Zero-Shot Sentiment Classification for {company}")
            return zeroshort_classifier.process_news(content, title, company=company)
        return None
    except Exception as e:
        print(f"Error in classification for {company}: {str(e)}")
        return None

def generate_summary_sync(processed_results):
    """Synchronous summary generation function"""
    try:
        if CLASSIFIER_TYPE == ClassifierType.FINBERT:
            print("Generating summary with FinBERT classifier")
            return finbert_classifier.generate_summary(processed_results)
        else:
            print("Generating summary with Zero-Shot classifier")
            return zeroshort_classifier.generate_summary(processed_results)
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return []


async def process_news_with_thread_pool(content: str, title: str, company: str):
    """Process news classification using thread pool"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(THREAD_POOL, classify_sync, content, title, company)



async def generate_summary_with_thread_pool(processed_results):
    """Generate summary using thread pool"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(THREAD_POOL, generate_summary_sync, processed_results)


async def scrape_and_analyze_news(request: ScrapedRequest):
    """
    Main business logic function to scrape and analyze news data for companies
    """
    try:
        start_time = time.time()
        logger.info(
            f"Received scrape request for: {request.name} and {request.company}")

        initial_search_queries = generate_initial_search_queries(request)
        search_results = []

        for each in initial_search_queries:
            search_results.extend(await duck_duck_go_searcher.search_text(each))
            search_results.extend(await duck_duck_go_searcher.search_news(each))

        search_results_urls = extract_urls_from_results(search_results)
        logger.info(
            f"Found {len(search_results_urls)} URLs to scrape for company discovery.")

        companies = set()
        async for result in scraper.scrape_urls_stream_immediate(search_results_urls):
            if result['status'] == 'success':
                content = result.get("content")
                if content:
                    output = await langchain_service.process_query_async(initial_search_queries, content)
                    print(output)
                    if output and output != "No companies found.":
                        found_companies = {
                            c.strip() for c in output.split(',')
                            if c.strip() and len(c.strip()) < 100
                        }
                        companies.update(found_companies)

        # Get unique companies
        companies_list = list(companies)
        companies_str = ", ".join(companies_list)
        unique_companies_str = await langchain_service.process_query_async_companies(
            "give me unique company names from the list", companies_str
        )

        # Convert to list
        unique_companies = [
            c.strip() for c in unique_companies_str.split(',')
            if c.strip()
        ]

        company_news_mapping = {}

        for company in unique_companies:
            logger.info(f"Found company: {company}")
            company_queries = generate_follow_up_queries(company)
            company_results = []

            for query in company_queries:
                results = await duck_duck_go_searcher.search_text(query)
                results.extend(await duck_duck_go_searcher.search_news(query))
                for result in results:
                    result['company'] = company  # Tag each result with its company
                company_results.extend(results)

            company_news_mapping[company] = company_results

        # Collect all news results
        news_search_results = []
        for results in company_news_mapping.values():
            news_search_results.extend(results)

        logger.info(f"Found {len(news_search_results)} news articles to scrape.")
        news_search_urls = extract_urls_from_results(news_search_results)

        classification_tasks = []

        async for result in scraper.scrape_urls_stream_immediate(news_search_urls):
            if result['status'] == 'success':
                content = result.get("content")
                title = result.get("title")
                company = None
                url = result.get("url")

                for comp, results_list in company_news_mapping.items():
                    for news_item in results_list:
                        item_url = (
                            news_item.get('href') or
                            news_item.get('link') or
                            news_item.get('url')
                        )
                        if item_url == url:
                            company = comp
                            break
                    if company:
                        break

                if content and title and company:
                    task = asyncio.create_task(
                        process_news_with_thread_pool(content, title, company)
                    )
                    classification_tasks.append(task)

        logger.info(
            f"Processing {len(classification_tasks)} articles in parallel threads")
        classification_results = await asyncio.gather(*classification_tasks, return_exceptions=True)

        processed_results = []
        for result in classification_results:
            if result is not None and not isinstance(result, Exception):
                processed_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Classification error: {str(result)}")

        # Generate summary
        summary = await generate_summary_with_thread_pool(processed_results)

        response_data = []
        for item in summary:
            response_data.append(ScrapedCompanyDetail(
                name=item["company_name"],
                average_sentiment=Sentiment(item["average_sentiment"]),
                negative_news_flag=item["negative_news_flag"],
                total_articles=item["total_articles"]
            ))

        execution_time = time.time() - start_time
        logger.info(
            f"Complete execution for scrape-and-flag in {execution_time} seconds")

        return ScrapedResponse(result=response_data)
    except Exception as e:
        logger.error(f"Error occurred during processing: {str(e)}")
        return ScrapedResponse(result=[])
