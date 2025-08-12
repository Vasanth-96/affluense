from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class ClassifierType(str, Enum):
    FINBERT = "finbert"
    ZEROSHORT = "zeroshort"
