import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
import os
from typing import List, Dict, Optional, Any
from pydantic import Field

from .base import BaseTool, ToolResult
from ..logger import logger
from ..config import config

# Determine a persistent path for ChromaDB data
# This should ideally be configurable or within a defined workspace data area
CHROMA_DB_PATH = os.path.join(config.workspace_root, ".chromadb_store")
COLLECTION_NAME = "pdf_documents"

# Attempt to get an embedding function based on LLM configuration
# This is a simplified example; robust selection/configuration is needed.
# It might try to use a configured OpenAI key or fallback to a sentence transformer.
try:
    if hasattr(config, 'llm_config') and \
       config.llm_config and \
       hasattr(config.llm_config, 'api_key') and config.llm_config.api_key and \
       hasattr(config.llm_config, 'base_url') and "openai" in config.llm_config.base_url.lower():
        # Safely get embedding_model, defaulting to "text-embedding-ada-002"
        openai_embedding_model = getattr(config.llm_config, 'embedding_model', "text-embedding-ada-002") or "text-embedding-ada-002"
        
        EMBEDDING_FUNCTION = embedding_functions.OpenAIEmbeddingFunction(
            api_key=config.llm_config.api_key,
            model_name=openai_embedding_model
        )
        logger.info(f"Using OpenAI for embeddings with model: {openai_embedding_model}.")
    else:
        EMBEDDING_FUNCTION = embedding_functions.DefaultEmbeddingFunction()
        logger.info("Using default SentenceTransformer (all-MiniLM-L6-v2) for embeddings.")
except Exception as e:
    logger.warning(f"Failed to initialize preferred embedding function: {e}. Falling back to default.")
    EMBEDDING_FUNCTION = embedding_functions.DefaultEmbeddingFunction()

class PDFSearchTool(BaseTool):
    name: str = "pdf_search"
    description: str = (
        "Searches for information within PDF documents. "
        "It can extract text from a PDF, (re)index its content for searching, "
        "and then perform a similarity search based on a user query."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "pdf_file_path": {
                "type": "string",
                "description": "The absolute or workspace-relative path to the PDF file.",
            },
            "query": {
                "type": "string",
                "description": "The search query to find relevant information in the PDF.",
            },
            "reindex": {
                "type": "boolean",
                "description": "(Optional) Whether to force re-indexing of the PDF content. Default is false.",
                "default": False,
            },
            "max_results": {
                "type": "integer",
                "description": "(Optional) Maximum number of search results to return. Default is 3.",
                "default": 3,
            }
        },
        "required": ["pdf_file_path", "query"],
    }

    chroma_client: Optional[chromadb.Client] = None
    collection: Optional[chromadb.Collection] = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._initialize_chroma()

    def _initialize_chroma(self):
        try:
            # Ensure the ChromaDB storage path exists
            os.makedirs(CHROMA_DB_PATH, exist_ok=True)
            logger.info(f"ChromaDB path: {CHROMA_DB_PATH}")

            self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            self.collection = self.chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=EMBEDDING_FUNCTION
            )
            logger.info(f"Successfully initialized ChromaDB and got/created collection '{COLLECTION_NAME}'.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}", exc_info=True)
            self.chroma_client = None
            self.collection = None

    def _document_is_indexed(self, pdf_file_path: str) -> bool:
        if not self.collection:
            return False
        try:
            results = self.collection.get(where={"source_pdf": pdf_file_path}, limit=1)
            return len(results["ids"]) > 0
        except Exception as e:
            logger.warning(f"Error checking if document '{pdf_file_path}' is indexed: {e}")
            return False

    def _extract_text_and_index(self, pdf_file_path: str) -> bool:
        if not self.collection:
            logger.error("ChromaDB collection not initialized. Cannot index PDF.")
            return False

        actual_path = pdf_file_path
        if not os.path.isabs(pdf_file_path):
            actual_path = os.path.join(config.workspace_root, pdf_file_path)

        if not os.path.exists(actual_path):
            logger.error(f"PDF file not found at: {actual_path}")
            return False

        try:
            logger.info(f"Starting text extraction and indexing for: {actual_path}")
            doc = fitz.open(actual_path)
            chunks = []
            chunk_ids = []
            metadatas = []
            
            # Simple chunking: one chunk per page
            # More sophisticated chunking (by paragraph, sentence, or fixed size with overlap)
            # would be better for retrieval quality.
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                if text.strip(): # Only add non-empty pages
                    chunks.append(text)
                    # Create a unique ID for each chunk
                    chunk_id = f"{os.path.basename(actual_path)}_page_{page_num + 1}"
                    chunk_ids.append(chunk_id)
                    metadatas.append({
                        "source_pdf": pdf_file_path, # Store original path for identification
                        "page_number": page_num + 1,
                        "document_name": os.path.basename(actual_path)
                    })
            
            doc.close()

            if not chunks:
                logger.warning(f"No text extracted from {actual_path}.")
                return True # No error, but nothing to index

            logger.info(f"Extracted {len(chunks)} text chunks from {actual_path}. Indexing...")
            
            # Remove existing entries for this PDF before re-indexing to avoid duplicates
            self.collection.delete(where={"source_pdf": pdf_file_path})
            
            self.collection.add(
                documents=chunks,
                ids=chunk_ids,
                metadatas=metadatas
            )
            logger.info(f"Successfully indexed {len(chunks)} chunks from {actual_path} into ChromaDB.")
            return True
        except Exception as e:
            logger.error(f"Error extracting text or indexing PDF '{actual_path}': {e}", exc_info=True)
            return False

    async def execute(
        self, pdf_file_path: str, query: str, reindex: bool = False, max_results: int = 3
    ) -> ToolResult:
        if not self.chroma_client or not self.collection:
            return ToolResult(
                stdout="",
                stderr="PDFSearchTool failed: ChromaDB not initialized.",
                result_json_str='{"error": "ChromaDB not initialized."}'
            )

        absolute_pdf_path = pdf_file_path
        if not os.path.isabs(pdf_file_path):
            absolute_pdf_path = os.path.join(config.workspace_root, pdf_file_path)
        
        if not os.path.exists(absolute_pdf_path):
             return ToolResult(
                stdout="",
                stderr=f"PDF file not found: {absolute_pdf_path}",
                result_json_str=f'{{"error": "PDF file not found: {absolute_pdf_path}"}}'
            )

        try:
            if reindex or not self._document_is_indexed(pdf_file_path):
                logger.info(f"Indexing required for '{pdf_file_path}'. Reindex flag: {reindex}")
                success = self._extract_text_and_index(pdf_file_path)
                if not success:
                    return ToolResult(
                        stdout="",
                        stderr=f"Failed to index PDF: {pdf_file_path}",
                        result_json_str=f'{{"error": "Failed to index PDF: {pdf_file_path}"}}'
                    )
            else:
                logger.info(f"PDF '{pdf_file_path}' is already indexed. Skipping indexing.")

            logger.info(f"Performing search in '{pdf_file_path}' for query: '{query}'")
            results = self.collection.query(
                query_texts=[query],
                n_results=max_results,
                where={"source_pdf": pdf_file_path} # Filter results to the specific PDF
            )
            
            if not results or not results.get("documents") or not results["documents"][0]:
                return ToolResult(
                    stdout=f"No relevant information found in '{pdf_file_path}' for query: '{query}'.",
                    stderr="",
                    result_json_str='{"results": [], "message": "No relevant information found."}'
                )

            # Format results
            output_str = f"Found the following relevant information in '{os.path.basename(pdf_file_path)}' for query '{query}':\n\n"
            search_results = []
            for i, doc_content in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {}
                page_num = metadata.get('page_number', 'N/A')
                entry = f"Source: Page {page_num}\nContent: {doc_content[:500]}...\n" # Truncate for brevity
                output_str += f"Result {i+1}:\n{entry}\n"
                search_results.append({
                    "content": doc_content,
                    "page_number": page_num,
                    "source_pdf": pdf_file_path,
                    "metadata": metadata,
                    "distance": results["distances"][0][i] if results["distances"] and results["distances"][0] else None
                })
            
            return ToolResult(
                stdout=output_str.strip(),
                stderr="",
                result_json_str=json.dumps({"results": search_results})
            )

        except Exception as e:
            logger.error(f"Error during PDF search for '{pdf_file_path}': {e}", exc_info=True)
            return ToolResult(
                stdout="",
                stderr=f"An unexpected error occurred during PDF search: {str(e)}",
                result_json_str=f'{{"error": "An unexpected error occurred: {str(e)}" }}'
            )

if __name__ == '__main__':
    # This is a simple test/example.
    # In a real scenario, the config would be loaded by the OpenManus framework.
    # For standalone testing, you might need to mock or set up a minimal config.
    
    # Mock a minimal config for testing if needed
    class MockLLMConfig:
        api_key: Optional[str] = os.environ.get("OPENAI_API_KEY") # Use env var for testing
        base_url: str = "https://api.openai.com/v1"
        embedding_model: Optional[str] = "text-embedding-ada-002"

    class MockConfig:
        workspace_root: str = os.path.abspath("./workspace_test_pdf") # Create this dir for testing
        llm_config: MockLLMConfig = MockLLMConfig()
        
    # Replace global config for this test
    global_config_backup = config
    config = MockConfig()
    os.makedirs(config.workspace_root, exist_ok=True)
    
    async def test_pdf_tool():
        # Create a dummy PDF for testing in config.workspace_root
        dummy_pdf_path = os.path.join(config.workspace_root, "test_document.pdf")
        try:
            doc = fitz.open() # New PDF
            page = doc.new_page()
            page.insert_text((50, 72), "Hello world! This is page 1 of a test PDF.")
            page.insert_text((50, 144), "Let's talk about apples and oranges.")
            page = doc.new_page()
            page.insert_text((50, 72), "This is page 2. We can discuss bananas here.")
            page.insert_text((50, 144), "The quick brown fox jumps over the lazy dog.")
            doc.save(dummy_pdf_path)
            doc.close()
            logger.info(f"Created dummy PDF: {dummy_pdf_path}")
        except Exception as e:
            logger.error(f"Could not create dummy PDF for testing: {e}")
            return

        tool = PDFSearchTool()

        if not tool.collection:
            logger.error("PDF Tool test: ChromaDB collection not initialized. Aborting test.")
            return

        # Test 1: Index and search
        logger.info("\n--- Test 1: Index and Search ---")
        result1 = await tool.execute(pdf_file_path="test_document.pdf", query="apples", reindex=True)
        print(f"STDOUT:\n{result1.stdout}")
        if result1.stderr: print(f"STDERR:\n{result1.stderr}")

        # Test 2: Search again (should use cache)
        logger.info("\n--- Test 2: Search Again (should be cached) ---")
        result2 = await tool.execute(pdf_file_path="test_document.pdf", query="bananas")
        print(f"STDOUT:\n{result2.stdout}")
        if result2.stderr: print(f"STDERR:\n{result2.stderr}")

        # Test 3: Search for something not there
        logger.info("\n--- Test 3: Search for non-existent term ---")
        result3 = await tool.execute(pdf_file_path="test_document.pdf", query="zebras")
        print(f"STDOUT:\n{result3.stdout}")
        if result3.stderr: print(f"STDERR:\n{result3.stderr}")
        
        # Test 4: Non-existent PDF
        logger.info("\n--- Test 4: Search in non-existent PDF ---")
        result4 = await tool.execute(pdf_file_path="non_existent.pdf", query="anything")
        print(f"STDOUT:\n{result4.stdout}")
        if result4.stderr: print(f"STDERR:\n{result4.stderr}")

        # Clean up dummy PDF and ChromaDB store
        try:
            # os.remove(dummy_pdf_path)
            # import shutil
            # if os.path.exists(CHROMA_DB_PATH):
            #     shutil.rmtree(CHROMA_DB_PATH)
            logger.info("Cleanup: Test PDF and ChromaDB store would be removed here in a real test teardown.")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
            
    # Restore global config
    config = global_config_backup

    # asyncio.run(test_pdf_tool()) # Requires async context, usually provided by agent framework
    # For direct execution:
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(test_pdf_tool())

    # logger.info("PDFSearchTool defined. Run 'python -m app.tool.pdf_search_tool' to test (requires setup).")
    # The above if __name__ == '__main__' block is for standalone testing and might need adjustments
    # to properly run within the OpenManus agent's async environment or if config isn't globally patched. 