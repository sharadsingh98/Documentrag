"""LangGraph nodes for RAG workflow + ReAct Agent inside generate_content"""

"""LangGraph nodes for RAG workflow + ReAct Agent inside generate_content"""

from uuid import uuid4
import sys
from typing import List, Optional

# Make uuid available globally for type hint resolution
sys.modules['__main__'].uuid4 = uuid4

"""LangGraph nodes for RAG workflow + ReAct Agent"""

from typing import List, Optional
from src.state.rag_state import RAGState
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None

    def retrieve_docs(self, state: RAGState) -> dict:
        """Classic retriever node"""
        docs = self.retriever.invoke(state.question)
        return {"retrieved_docs": docs}

    def _build_tools(self):
        """Build retriever + wikipedia tools"""

        # Use closure to capture self.retriever
        retriever_ref = self.retriever

        @tool
        def retriever_search(query: str) -> str:
            """Fetch passages from indexed corpus about the user's documents."""
            try:
                docs = retriever_ref.invoke(query)
                if not docs:
                    return "No documents found."
                merged = []
                for i, d in enumerate(docs[:8], start=1):
                    meta = getattr(d, "metadata", {})
                    title = meta.get("title") or meta.get("source") or f"doc_{i}"
                    merged.append(f"[{i}] {title}\n{d.page_content}")
                return "\n\n".join(merged)
            except Exception as e:
                return f"Error retrieving documents: {str(e)}"

        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="en")
        )

        return [retriever_search, wiki]

    def _build_agent(self):
        """ReAct agent with tools"""
        tools = self._build_tools()
        system_prompt = (
            "You are a helpful RAG agent. "
            "Prefer 'retriever_search' for user-provided docs; use 'wikipedia' for general knowledge. "
            "Return only the final useful answer."
        )
        # Remove the prompt parameter - it's not supported
        self._agent = create_react_agent(self.llm, tools=tools)

    def generate_answer(self, state: RAGState) -> dict:
        """Generate answer using ReAct agent with retriever + wikipedia."""
        if self._agent is None:
            self._build_agent()

        result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})

        messages = result.get("messages", [])
        answer = None
        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)

        return {"answer": answer or "Could not generate answer."}