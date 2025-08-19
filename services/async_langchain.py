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
                "You are an expert financial analyst specializing in identifying verified business relationships between high-net-worth individuals and companies.\n\n"
                "QUERY (Target person and context): {query}\n\n"
                "TEXT TO ANALYZE:\n{text}\n\n"
                "INSTRUCTIONS:\n"
                "1. RELEVANCE CHECK: First verify this text discusses the specific person mentioned in the QUERY.\n"
                "2. EVIDENCE REQUIREMENT: Extract company names ONLY when the text provides EXPLICIT evidence of:\n"
                "   • Ownership (founder, co-founder, owner, shareholder, stake holder)\n"
                "   • Executive leadership (CEO, Chairman, President, Managing Director, CTO, CFO)\n"
                "   • Board positions (Board member, Director, Board Chairman)\n"
                "   • Control relationships (subsidiary ownership, parent company control)\n"
                "   • Business partnerships (joint ventures, business partners, co-investors)\n"
                "   • Investment relationships (investor, backed by, funded by)\n\n"
                "3. EXCLUSION CRITERIA - Do NOT include companies if:\n"
                "   • Person is only mentioned as a customer, user, or client\n"
                "   • Company is mentioned in passing without connection details\n"
                "   • Person is only an employee without leadership role\n"
                "   • Connection is speculative, rumored, or unconfirmed\n"
                "   • Text only mentions industry competitors or market comparisons\n"
                "   • Historical mentions without current relevance\n\n"
                "4. VERIFICATION: For each company, ensure there's a clear sentence or phrase establishing the connection.\n\n"
                "5. FORMATTING: Use complete, formal company names (include 'Inc', 'Ltd', 'Corporation', 'Limited' when mentioned).\n\n"
                "6. OUTPUT FORMAT:\n"
                "   • If companies found: Comma-separated list only (no bullets, numbers, or explanations)\n"
                "   • If no qualifying companies: Exactly 'No companies found.'\n\n"
                "EXAMPLES:\n"
                "Good: 'Tesla Inc, SpaceX, Neuralink Corporation'\n"
                "Bad: '1. Tesla Inc (CEO), 2. SpaceX (Founder)'\n\n"
                "OUTPUT:"
            ),
        )

        # Deduplication/cleanup prompt
        self.unique_companies_template = PromptTemplate(
            input_variables=["query", "text"],
            template=(
                "You are a data-cleaning specialist for company name standardization.\n\n"
                "QUERY (person and context): {query}\n\n"
                "RAW COMPANY LIST:\n{text}\n\n"
                "CLEANING INSTRUCTIONS:\n"
                "1. DUPLICATE REMOVAL:\n"
                "   • Merge variations of the same company (different spellings, capitalizations, abbreviations)\n"
                "   • Choose the most complete/formal version as the canonical name\n\n"
                "2. INVALID ENTRIES - Remove:\n"
                "   • Non-company entries (locations, projects, mines, properties)\n"
                "   • Incomplete/unclear names (single words without context)\n"
                "   • Personal descriptions in parentheses (e.g., 'Company (Co-founder)')\n"
                "   • Generic business terms without specific company identification\n\n"
                "3. STANDARDIZATION:\n"
                "   • Keep official suffixes: Inc, Ltd, LLC, Corporation, Limited, Private Limited, Pvt Ltd\n"
                "   • Use proper capitalization\n"
                "   • Remove extra spaces and punctuation\n"
                "   • Prefer full legal names over abbreviated versions\n\n"
                "4. EXAMPLES OF MERGING:\n"
                "   • 'BluSmart', 'BluSmart Mobility', 'BluSmart Mobility Pvt Ltd' → 'BluSmart Mobility Pvt Ltd'\n"
                "   • 'Microsoft Corp', 'Microsoft Corporation' → 'Microsoft Corporation'\n"
                "   • 'Tata Motors', 'Tata Motors Limited' → 'Tata Motors Limited'\n\n"
                "5. EXAMPLES OF REMOVAL:\n"
                "   • 'Gurugram Properties Ltd' (if it's a location/property, not a company)\n"
                "   • 'Gualcamayo gold mine' (mining project, not company)\n"
                "   • 'Bennett' (incomplete name)\n"
                "   • 'Bluesmart (Co-founder)' → Remove parenthetical, keep 'BluSmart'\n\n"
                "6. OUTPUT FORMAT:\n"
                "   • Comma-separated list only\n"
                "   • No numbering, bullets, or explanations\n"
                "   • If no valid companies: 'No companies found.'\n\n"
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
