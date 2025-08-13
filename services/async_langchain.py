from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

class LangChainAsyncService:
    def __init__(self, model: str = "mistral:latest", temperature: float = 0.0) -> None:
        self.llm = OllamaLLM(model=model, temperature=temperature)

        # Extraction prompt
        self.prompt_template = PromptTemplate(
            input_variables=["query", "text"],
            template=(
                "You are an expert assistant in identifying companies with confirmed connections to high-net-worth individuals.\n"
                "QUERY (person's name and possibly one known company): {query}\n\n"
                "TEXT (context to analyze):\n{text}\n\n"
                "TASK:\n"
                "1. Determine if the text is relevant to the person in QUERY.\n"
                "2. Extract ONLY company names where there is clear, explicit, or strongly implied evidence that the person:\n"
                "   - Founded, co-founded, or started the company.\n"
                "   - Owns, partly owns, or has a significant stake in the company.\n"
                "   - Is a partner, chairman, CEO, board member, or holds a leadership role.\n"
                "   - Directly controls a subsidiary or parent company.\n"
                "   - Is in a formal joint venture or business partnership with the company.\n"
                "3. Ignore companies if the text only mentions them in passing without evidence of a formal or ownership connection.\n"
                "4. Normalize company names to their most complete, formal version (e.g., 'Tesla' → 'Tesla Inc').\n"
                "5. Output exactly a comma-separated list of company names without numbering, bullet points, or extra words.\n"
                "6. If no qualifying companies are found, output exactly: No companies found.\n\n"
                "FORMAT EXAMPLE:\n"
                "Microsoft Corporation, Apple Inc, Tesla Inc\n\n"
                "OUTPUT:"
            ),
        )

        # Deduplication/cleanup prompt
        self.unique_companies_template = PromptTemplate(
            input_variables=["query", "text"],
            template=(
                "You are a data-cleaning assistant specializing in company names.\n"
                "QUERY (person and possible known company): {query}\n\n"
                "RAW COMPANY LIST:\n{text}\n\n"
                "TASK:\n"
                "1. Remove duplicates (treat names with different capitalization, spacing, or punctuation as the same or any other similar as same).\n"
                "2. Merge synonyms/abbreviations into the most complete, formal company name available.\n"
                "3. Keep correct spelling and retain company type suffixes (Inc., Ltd., LLC, etc.) when available.\n"
                "4. Remove leading/trailing spaces.\n"
                "5. Output exactly a comma-separated list with no numbering, bullets, or extra text.\n"
                "6. If no valid companies remain, output exactly: No companies found.\n\n"
                "FORMAT EXAMPLE:\n"
                "Microsoft Corporation, Apple Inc, Tesla Inc\n\n"
                "OUTPUT:"
            ),
        )


        self.chain = self.prompt_template | self.llm | StrOutputParser()
        self.companies_chain = self.unique_companies_template | self.llm | StrOutputParser()

    async def process_query_async(self, query: str, text: str) -> str:
        """Extract company names from raw text."""
        try:
            result = await self.chain.ainvoke({"query": query, "text": text})
            return result.strip()
        except Exception as e:
            print(f"Error in process_query_async: {str(e)}")
            return "No companies found."

    async def process_query_async_companies(self, query: str, text: str) -> str:
        """Deduplicate/clean a company list."""
        try:
            result = await self.companies_chain.ainvoke({"query": query, "text": text})
            return result.strip()
        except Exception as e:
            print(f"Error in process_query_async_companies: {str(e)}")
            return text  # Fallback to original list

langchain_service = LangChainAsyncService()
