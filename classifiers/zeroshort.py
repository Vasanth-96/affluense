from collections import defaultdict
from transformers import pipeline
import logging
from utils.enums import Sentiment
import threading
from typing import Dict, List
# Set up basic logging for sync operations
logging.basicConfig(level=logging.INFO)
sync_logger = logging.getLogger(__name__)

NEGATIVE_KEYWORDS = [
    "fraud", "scam", "lawsuit", "corruption", "controversy",
    "penalty", "fine", "arrest", "investigation", "illegal"
]

SENTIMENT_SCORES = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}

class ZeroShotSentimentProcessor:
    def __init__(self):
        self.classifier = None
        self.candidate_labels = ["positive", "negative", "neutral"]
        self._init_lock = threading.Lock()
        self._initialized = False
    
    def _ensure_classifier(self):
        """Ensure classifier is initialized (thread-safe)"""
        if not self._initialized:
            with self._init_lock:
                if not self._initialized:  # Double-check pattern
                    try:
                        sync_logger.info(f"Initializing Zero-Shot classifier for thread: {threading.current_thread().name}")
                        self.classifier = pipeline(
                            "zero-shot-classification", 
                            model="facebook/bart-large-mnli",
                            device=-1,
                            return_all_scores=True
                        )
                        self._initialized = True
                        sync_logger.info("Zero-Shot classifier initialized successfully")
                    except Exception as e:
                        sync_logger.error(f"Failed to initialize Zero-Shot classifier: {str(e)}")
                        self.classifier = None

    def predict(self, text: str) -> str:
        """Predict sentiment using zero-shot classification"""
        if not text or not text.strip():
            return "neutral"
        
        self._ensure_classifier()
        if self.classifier is None:
            sync_logger.warning("Zero-Shot classifier not available, returning neutral")
            return "neutral"
        
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]
        
        try:
            result = self.classifier(text, self.candidate_labels)
            return result['labels'][0] if result and result.get('labels') else "neutral"
        except Exception as e:
            sync_logger.error(f"Error in Zero-Shot sentiment prediction: {str(e)}")
            return "neutral"
    
    def predict_batch(self, texts: List[str]) -> List[str]:
        """Predict sentiment for multiple texts in batch"""
        if not texts:
            return []
        
        self._ensure_classifier()
        if self.classifier is None:
            return ["neutral"] * len(texts)
        
        # Truncate texts to max length
        max_length = 512
        processed_texts = [text[:max_length] if text and len(text) > max_length 
                        else (text or "") for text in texts]
        
        try:
            # Process all texts in a single batch
            results = []
            for text in processed_texts:
                if not text.strip():
                    results.append("neutral")
                else:
                    result = self.classifier(text, self.candidate_labels)
                    results.append(result['labels'][0] if result and result.get('labels') else "neutral")
            return results
        except Exception as e:
            sync_logger.error(f"Error in batch Zero-Shot prediction: {str(e)}")
            return ["neutral"] * len(texts)

    def process_news_batch(self, news_items: List[Dict]) -> List[Dict]:
        """Process multiple news items in batch"""
        if not news_items:
            return []
        
        texts = []
        for item in news_items:
            title = item.get('title', '')
            content = item.get('content', '')
            combined_text = f"{title}. {content}" if title else content
            texts.append(combined_text)
        
        sentiments = self.predict_batch(texts)
        
        results = []
        for i, (item, sentiment) in enumerate(zip(news_items, sentiments)):
            try:
                company = item.get('company')
                title = item.get('title', '')
                content = item.get('content', '')
                combined_text = texts[i]
                
                negative_flag = self.flag_negative_news(combined_text)
                
                results.append({
                    "company_name": str(company) if company else "",
                    "title": str(title),
                    "content": str(content),
                    "sentiment": sentiment,
                    "sentiment_score": SENTIMENT_SCORES.get(sentiment, 0),
                    "negative_news_flag": negative_flag
                })
            except Exception as e:
                sync_logger.error(f"Error processing news item {i}: {str(e)}")
                continue
        
        return results


    def flag_negative_news(self, text: str) -> bool:
        """Check if text contains negative keywords"""
        if not text:
            return False
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in NEGATIVE_KEYWORDS)

    def average_sentiment_label(self, avg_score: float) -> str:
        """Convert average score to sentiment label matching the enum"""
        if avg_score > 0.3:
            return Sentiment.POSITIVE.value  # Returns "positive"
        elif avg_score < -0.3:
            return Sentiment.NEGATIVE.value  # Returns "negative"
        else:
            return Sentiment.NEUTRAL.value   # Returns "neutral"

    def process_news(self, content: str, title: str, company: str = None):
        """
        Process a single news item (thread-safe)
        Returns: Dict with sentiment analysis results
        """
        try:
            if not company:
                sync_logger.warning("No company provided for news processing")
                return None
                
            combined_text = f"{title}. {content}" if title else content
            
            sentiment = self.predict(combined_text)
            negative_flag = self.flag_negative_news(combined_text)

            return {
                "company_name": str(company),
                "title": str(title) if title else "",
                "content": str(content) if content else "",
                "sentiment": sentiment,
                "sentiment_score": SENTIMENT_SCORES.get(sentiment, 0),
                "negative_news_flag": negative_flag
            }
        except Exception as e:
            sync_logger.error(f"Error processing news for {company}: {str(e)}")
            return None

    def generate_summary(self, processed_results):
        """
        Generate summary from processed results (thread-safe)
        processed_results: List of results from process_news
        Returns: List of company summaries
        """
        if not processed_results:
            sync_logger.warning("No processed results to generate summary from")
            return []
        
        try:
            company_sentiments = defaultdict(list)
            company_negative_flags = defaultdict(bool)

            for result in processed_results:
                if not result or not result.get('company_name'):
                    continue

                company = result['company_name']
                sentiment_score = result.get('sentiment_score', 0)
                negative_flag = result.get('negative_news_flag', False)
                
                company_sentiments[company].append(sentiment_score)
                company_negative_flags[company] = company_negative_flags[company] or negative_flag

            summary = []
            for company, scores in company_sentiments.items():
                if not scores:
                    continue
                    
                avg_score = sum(scores) / len(scores)
                avg_sentiment = self.average_sentiment_label(avg_score)
                negative_flag = company_negative_flags.get(company, False)

                summary.append({
                    "company_name": company,
                    "average_sentiment": avg_sentiment,  # Returns enum value
                    "negative_news_flag": negative_flag,
                    "total_articles": len(scores)
                })

            sync_logger.info(f"Generated Zero-Shot summary for {len(summary)} companies")
            return summary
            
        except Exception as e:
            sync_logger.error(f"Error generating Zero-Shot summary: {str(e)}")
            return []

    def cleanup(self):
        """Clean up classifier to free memory"""
        try:
            if self.classifier is not None:
                # Clear the classifier
                self.classifier = None
                self._initialized = False
                sync_logger.info(f"Cleaned up Zero-Shot classifier for thread: {threading.current_thread().name}")
        except Exception as e:
            sync_logger.error(f"Error during cleanup: {str(e)}")

    def get_status(self):
        """Get current status of the classifier"""
        return {
            "initialized": self._initialized,
            "classifier_available": self.classifier is not None,
            "thread": threading.current_thread().name
        }

zeroshort_classifier = ZeroShotSentimentProcessor()
