import json
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import torch.nn.functional as F
from utils.sync_logger import sync_logger as logger
from utils.enums import Sentiment
import threading
from typing import Dict, List


NEGATIVE_KEYWORDS = [
    "fraud", "scam", "lawsuit", "corruption", "controversy",
    "penalty", "fine", "arrest", "investigation", "illegal"
]


SENTIMENT_SCORES = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}


class FinBertSentimentProcessor:
    def __init__(self):
        self.classifier = None
        self._init_lock = threading.Lock()
        self._initialized = False
        self.model_name = "yiyanghkust/finbert-tone"
        self.label_map = {0: "neutral", 1: "positive", 2: "negative"}
    
    def _ensure_classifier(self):
        """Ensure classifier is initialized (thread-safe) - Similar to ZeroShot pattern"""
        if not self._initialized:
            with self._init_lock:
                if not self._initialized:  # Proper double-check pattern
                    try:
                        logger.info(f"Initializing FinBERT classifier for thread: {threading.current_thread().name}")
                        self.classifier = pipeline(
                            "text-classification",
                            model=self.model_name,
                            device=-1,  # Force CPU for thread safety
                            return_all_scores=True
                        )
                        self._initialized = True
                        logger.info("FinBERT classifier initialized successfully")
                    except Exception as e:
                        logger.error(f"Failed to initialize FinBERT classifier: {str(e)}")
                        self.classifier = None


    def predict(self, text: str) -> str:
        """Predict sentiment using FinBERT"""
        if not text or not text.strip():
            return "neutral"

        self._ensure_classifier()
        if self.classifier is None:
            logger.warning("FinBERT classifier not available, returning neutral")
            return "neutral"

        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]

        try:
            result = self.classifier(text)
            if result and len(result) > 0:
                # Get the label with highest score
                best_result = max(result, key=lambda x: x['score'])
                label = best_result['label'].lower()
                
                # Map FinBERT labels to standard format
                if 'positive' in label:
                    return "positive"
                elif 'negative' in label:
                    return "negative"
                else:
                    return "neutral"
            
            return "neutral"

        except Exception as e:
            logger.error(f"Error in FinBERT sentiment prediction: {str(e)}")
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
            # FinBERT pipeline can handle batch processing
            results = []
            
            # Process in smaller batches to avoid memory issues
            batch_size = 16  # Adjust based on your memory constraints
            for i in range(0, len(processed_texts), batch_size):
                batch_texts = processed_texts[i:i + batch_size]
                
                # Filter empty texts
                valid_texts = [text for text in batch_texts if text.strip()]
                if not valid_texts:
                    results.extend(["neutral"] * len(batch_texts))
                    continue
                
                try:
                    batch_results = self.classifier(valid_texts)
                    
                    # Process results
                    valid_idx = 0
                    for text in batch_texts:
                        if not text.strip():
                            results.append("neutral")
                        else:
                            if valid_idx < len(batch_results):
                                result = batch_results[valid_idx]
                                if isinstance(result, list) and result:
                                    # Get the label with highest score
                                    best_result = max(result, key=lambda x: x['score'])
                                    label = best_result['label'].lower()
                                    
                                    if 'positive' in label:
                                        results.append("positive")
                                    elif 'negative' in label:
                                        results.append("negative")
                                    else:
                                        results.append("neutral")
                                else:
                                    results.append("neutral")
                                valid_idx += 1
                            else:
                                results.append("neutral")
                                
                except Exception as batch_e:
                    logger.error(f"Error in batch processing: {str(batch_e)}")
                    results.extend(["neutral"] * len(batch_texts))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch FinBERT prediction: {str(e)}")
            return ["neutral"] * len(texts)

    def process_news_batch(self, news_items: List[Dict]) -> List[Dict]:
        """Process multiple news items in batch"""
        if not news_items:
            return []
        
        # Prepare texts for batch processing
        texts = []
        for item in news_items:
            title = item.get('title', '')
            content = item.get('content', '')
            combined_text = f"{title}. {content}" if title else content
            texts.append(combined_text)
        
        # Get batch predictions
        sentiments = self.predict_batch(texts)
        
        # Process results
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
                logger.error(f"Error processing news item {i}: {str(e)}")
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
            return Sentiment.POSITIVE.value
        elif avg_score < -0.3:
            return Sentiment.NEGATIVE.value
        else:
            return Sentiment.NEUTRAL.value

    def process_news(self, content: str, title: str, company: str = None):
        """Process a single news item (thread-safe)"""
        try:
            if not company:
                logger.warning("No company provided for news processing")
                return None
                
            # Combine title and content
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
            logger.error(f"Error processing news with FinBERT for {company}: {str(e)}")
            return None

    def generate_summary(self, processed_results):
        """Generate summary from processed results (thread-safe)"""
        if not processed_results:
            logger.warning("No processed results to generate summary from")
            return []

        try:
            company_sentiments = defaultdict(list)
            company_negative_flags = defaultdict(bool)

            # Group by company
            for result in processed_results:
                if not result or not result.get('company_name'):
                    continue

                company = result['company_name']
                sentiment_score = result.get('sentiment_score', 0)
                negative_flag = result.get('negative_news_flag', False)

                company_sentiments[company].append(sentiment_score)
                company_negative_flags[company] = company_negative_flags[company] or negative_flag

            # Generate summaries
            summary = []
            for company, scores in company_sentiments.items():
                if not scores:
                    continue

                avg_score = sum(scores) / len(scores)
                avg_sentiment = self.average_sentiment_label(avg_score)
                negative_flag = company_negative_flags.get(company, False)

                summary.append({
                    "company_name": company,
                    "average_sentiment": avg_sentiment,
                    "negative_news_flag": negative_flag,
                    "total_articles": len(scores)
                })

            logger.info(f"Generated FinBERT summary for {len(summary)} companies")
            return summary

        except Exception as e:
            logger.error(f"Error generating FinBERT summary: {str(e)}")
            return []

    def cleanup(self):
        """Clean up classifier to free memory"""
        try:
            if self.classifier is not None:
                self.classifier = None
                self._initialized = False
                logger.info(f"Cleaned up FinBERT classifier for thread: {threading.current_thread().name}")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def get_status(self):
        """Get current status of the classifier"""
        return {
            "initialized": self._initialized,
            "classifier_available": self.classifier is not None,
            "thread": threading.current_thread().name
        }


finbert_classifier = FinBertSentimentProcessor()
