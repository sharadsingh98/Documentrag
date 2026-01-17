"""Graph builder for LangGraph workflow"""

from typing import Optional
from uuid import uuid4
from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.node.reactnode import RAGNodes


class GraphBuilder:
    """Builds and manages the LangGraph workflow for RAG operations"""
    
    def __init__(self, retriever, llm):
        """
        Initialize graph builder
        
        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        self.retriever = retriever
        self.llm = llm
        self.nodes = RAGNodes(retriever, llm)
        self.graph: Optional[StateGraph] = None
    
    def build(self) -> StateGraph:
        """
        Build the RAG workflow graph
        
        Returns:
            Compiled graph instance
        """
        # Create state graph
        builder = StateGraph(RAGState)
        
        # Add nodes
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("responder", self.nodes.generate_answer)
        
        # Set entry point
        builder.set_entry_point("retriever")
        
        # Add edges
        builder.add_edge("retriever", "responder")
        builder.add_edge("responder", END)
        
        # Compile and cache graph
        self.graph = builder.compile()
        return self.graph
    
    def run(self, question: str) -> dict:
        """
        Run the RAG workflow with the given question
        
        Args:
            question: User question to answer
            
        Returns:
            Final state dictionary containing the answer and metadata
            
        Raises:
            ValueError: If question is empty or None
        """
        # Validate input
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        # Build graph if not already built
        if self.graph is None:
            self.build()
        
        # Create initial state
        initial_state = RAGState(question=question.strip())
        
        # Run workflow
        result = self.graph.invoke(initial_state)
        return result
    
    def reset(self):
        """Reset the graph, forcing rebuild on next run"""
        self.graph = None
    
    def __repr__(self) -> str:
        """String representation of GraphBuilder"""
        status = "built" if self.graph is not None else "not built"
        return f"GraphBuilder(status={status})"