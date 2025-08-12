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
                "You are an assistant that extracts company names from text.\n"
                "QUERY: {query}\n\n"
                "TEXT:\n{text}\n\n"
                "TASK:\n"
                "- Identify ONLY companies directly associated with the person in the query.\n"
                "- Output EXACTLY a comma-separated list of company names.\n"
                "- No numbering, bullet points, or extra words.\n"
                "- If none found, output exactly: No companies found.\n\n"
                "FORMAT EXAMPLE:\n"
                "Microsoft Corporation, Apple Inc, Tesla\n\n"
                "OUTPUT:"
            ),
        )

        # Deduplication/cleanup prompt
        self.unique_companies_template = PromptTemplate(
            input_variables=["query", "text"],
            template=(
                "You are given a list of company names. Clean and deduplicate them.\n"
                "QUERY: {query}\n\n"
                "COMPANIES:\n{text}\n\n"
                "TASK:\n"
                "- Remove duplicates.\n"
                "- Keep correct spelling and full names.\n"
                "- Output EXACTLY a comma-separated list.\n"
                "- If none remain, output exactly: No companies found.\n\n"
                "FORMAT EXAMPLE:\n"
                "Microsoft Corporation, Apple Inc, Tesla\n\n"
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
