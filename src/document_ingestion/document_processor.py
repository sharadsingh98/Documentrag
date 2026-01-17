"""Document processing module for loading and splitting documents"""

import logging
from pathlib import Path
from typing import List, Union
from uuid import uuid4

from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document loading and processing with support for URLs, PDFs, and text files"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document processor

        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def load_from_url(self, url: str) -> List[Document]:
        """
        Load document(s) from a URL

        Args:
            url: Web URL to load content from

        Returns:
            List of loaded documents

        Raises:
            ValueError: If URL is invalid or cannot be loaded
        """
        try:
            logger.info(f"Loading document from URL: {url}")
            loader = WebBaseLoader(url)
            docs = loader.load()
            logger.info(f"Successfully loaded {len(docs)} document(s) from URL")
            return docs
        except Exception as e:
            logger.error(f"Failed to load URL {url}: {str(e)}")
            raise ValueError(f"Failed to load URL {url}: {str(e)}")

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """
        Load documents from all PDFs in a directory

        Args:
            directory: Path to directory containing PDF files

        Returns:
            List of loaded documents from all PDFs

        Raises:
            ValueError: If directory doesn't exist or contains no PDFs
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        try:
            logger.info(f"Loading PDFs from directory: {directory}")
            loader = PyPDFDirectoryLoader(str(dir_path))
            docs = loader.load()
            
            if not docs:
                logger.warning(f"No PDF documents found in directory: {directory}")
            else:
                logger.info(f"Successfully loaded {len(docs)} document(s) from PDF directory")
            
            return docs
        except Exception as e:
            logger.error(f"Failed to load PDFs from directory {directory}: {str(e)}")
            raise ValueError(f"Failed to load PDFs from directory {directory}: {str(e)}")

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load document(s) from a single PDF file

        Args:
            file_path: Path to the PDF file

        Returns:
            List of loaded documents (one per page)

        Raises:
            ValueError: If file doesn't exist or cannot be loaded
        """
        pdf_path = Path(file_path)
        
        if not pdf_path.exists():
            raise ValueError(f"PDF file does not exist: {file_path}")
        
        if pdf_path.suffix.lower() != '.pdf':
            raise ValueError(f"File is not a PDF: {file_path}")

        try:
            logger.info(f"Loading PDF file: {file_path}")
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            logger.info(f"Successfully loaded {len(docs)} page(s) from PDF")
            return docs
        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {str(e)}")
            raise ValueError(f"Failed to load PDF {file_path}: {str(e)}")

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load document(s) from a text file

        Args:
            file_path: Path to the text file

        Returns:
            List containing the loaded document

        Raises:
            ValueError: If file doesn't exist or cannot be loaded
        """
        txt_path = Path(file_path)
        
        if not txt_path.exists():
            raise ValueError(f"Text file does not exist: {file_path}")

        try:
            logger.info(f"Loading text file: {file_path}")
            loader = TextLoader(str(txt_path), encoding="utf-8")
            docs = loader.load()
            logger.info(f"Successfully loaded text file")
            return docs
        except Exception as e:
            logger.error(f"Failed to load text file {file_path}: {str(e)}")
            raise ValueError(f"Failed to load text file {file_path}: {str(e)}")

    def load_documents(self, sources: List[str]) -> List[Document]:
        """
        Load documents from multiple sources (URLs, PDF files/directories, or text files)

        Args:
            sources: List of source paths or URLs

        Returns:
            List of all loaded documents

        Raises:
            ValueError: If any source is invalid or unsupported
        """
        if not sources:
            logger.warning("No sources provided to load_documents")
            return []

        all_docs: List[Document] = []
        
        for src in sources:
            try:
                # Check if it's a URL
                if src.startswith("http://") or src.startswith("https://"):
                    docs = self.load_from_url(src)
                    all_docs.extend(docs)
                else:
                    # It's a file path
                    path = Path(src)
                    
                    if not path.exists():
                        logger.error(f"Path does not exist: {src}")
                        raise ValueError(f"Path does not exist: {src}")
                    
                    if path.is_dir():
                        # PDF directory
                        docs = self.load_from_pdf_dir(path)
                        all_docs.extend(docs)
                    elif path.suffix.lower() == ".pdf":
                        # Single PDF file
                        docs = self.load_from_pdf(path)
                        all_docs.extend(docs)
                    elif path.suffix.lower() == ".txt":
                        # Text file
                        docs = self.load_from_txt(path)
                        all_docs.extend(docs)
                    else:
                        raise ValueError(
                            f"Unsupported file type: {src}. "
                            "Supported types: .pdf, .txt, or directories containing PDFs"
                        )
            except Exception as e:
                logger.error(f"Error processing source {src}: {str(e)}")
                raise

        logger.info(f"Total documents loaded: {len(all_docs)}")
        return all_docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks

        Args:
            documents: List of documents to split

        Returns:
            List of document chunks
        """
        if not documents:
            logger.warning("No documents provided to split_documents")
            return []

        logger.info(f"Splitting {len(documents)} document(s) into chunks")
        chunks = self.splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunk(s)")
        return chunks

    def process_urls(self, urls: List[str]) -> List[Document]:
        """
        Complete pipeline: load documents from sources and split into chunks

        Args:
            urls: List of source paths or URLs (despite the name, supports files too)

        Returns:
            List of processed document chunks
        """
        logger.info(f"Starting document processing pipeline for {len(urls)} source(s)")
        docs = self.load_documents(urls)
        chunks = self.split_documents(docs)
        logger.info("Document processing pipeline completed")
        return chunks

    def add_metadata(self, documents: List[Document], metadata: dict) -> List[Document]:
        """
        Add custom metadata to all documents

        Args:
            documents: List of documents to add metadata to
            metadata: Dictionary of metadata to add

        Returns:
            List of documents with updated metadata
        """
        for doc in documents:
            doc.metadata.update(metadata)
        return documents

    def filter_documents(
        self, documents: List[Document], min_length: int = 0
    ) -> List[Document]:
        """
        Filter documents by minimum content length

        Args:
            documents: List of documents to filter
            min_length: Minimum character length to keep

        Returns:
            Filtered list of documents
        """
        filtered = [doc for doc in documents if len(doc.page_content) >= min_length]
        logger.info(
            f"Filtered documents: {len(documents)} -> {len(filtered)} "
            f"(removed {len(documents) - len(filtered)})"
        )
        return filtered


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)

    # Example sources
    sources = [
        "https://example.com/article",
        "path/to/documents",  # PDF directory
        "path/to/document.pdf",  # Single PDF
        "path/to/document.txt",  # Text file
    ]

    try:
        # Process documents
        chunks = processor.process_urls(sources)
        print(f"Processed {len(chunks)} document chunks")
    except Exception as e:
        print(f"Error processing documents: {e}")