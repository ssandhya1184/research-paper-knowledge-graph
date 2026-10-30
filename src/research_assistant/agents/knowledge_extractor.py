
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from research_assistant.models import ResearchExtraction

load_dotenv()


def create_llm() -> ChatGoogleGenerativeAI:
    """Create a Gemini client."""

    api_key = os.getenv("GOOGLE_API_KEY")
    #model=os.getenv("GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
    model = "gemini-3.5-flash-lite"
    
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing from .env")

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0,
    )


def extract_knowledge(paper_text: str) -> ResearchExtraction:
    """Extract research entities and relationships."""

    llm = create_llm().with_structured_output(
        ResearchExtraction,
        method="json_schema",
    )

    prompt = f"""
    Analyze the scientific research paper below.

    Extract methods, datasets, evaluation metrics, concepts,
    key findings, and relationships between entities.

    Use only information supported by the paper.
    Do not invent entities or relationships.
    Use consistent entity names in relationships.
    Return empty lists when information is unavailable.

    PAPER TEXT:
    {paper_text[:20000]}
    """

    return llm.invoke(prompt)
