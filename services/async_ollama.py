from langchain_ollama import OllamaLLM
from typing import List, Any, Dict, AsyncIterator
from langchain_core.runnables import RunnableConfig

class OllamaAsyncService:
    def __init__(self, model: str = "mistral:latest", temperature: float = 0.7):
        self.llm = OllamaLLM(model=model, temperature=temperature)

    async def generate_async(self, prompt: str) -> Any:
        return await self.llm.ainvoke(prompt)

    async def generate_batch_async(self, prompts: List[str]) -> Any:
        return await self.llm.agenerate(prompts)

    async def stream_response(self, prompt: str, config: RunnableConfig = None) -> AsyncIterator[Any]:
        async for chunk in self.llm.astream(prompt, config=config):
            yield chunk
