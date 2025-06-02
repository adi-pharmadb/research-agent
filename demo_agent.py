import asyncio
import os
# import logging # Removed standard logging

# Adjust the python path to include openmanus_core if running from root
import sys

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.dirname(__file__))
openmanus_core_path = os.path.join(project_root, "openmanus_core")
# Add openmanus_core to the Python path
if openmanus_core_path not in sys.path:
    sys.path.insert(0, openmanus_core_path)

from app.agent.manus import Manus
from app.config import config # This will load config.toml
from app.logger import logger # Use the app's logger (Loguru)

# Configure logger for the demo (Loguru)
# We'll rely on Loguru's default INFO level to stderr.

# --- Sample Data Paths ---
SAMPLE_CSV_PATH = os.path.join(openmanus_core_path, "workspace", "sample_data.csv")
SAMPLE_PDF_PATH = os.path.join(openmanus_core_path, "workspace", "sample_doc.pdf")

# --- Prompts for the Agent ---
PROMPTS = [
    {
        "id": "pdf_query",
        "description": "Querying the sample PDF document.",
        "text": f"What is the document '{SAMPLE_PDF_PATH}' about? Summarize its key points."
    },
    #{
    #    "id": "pdf_reindex_query",
    #    "description": "Querying the sample PDF document with re-indexing.",
    #    "text": f"Tell me more details from '{SAMPLE_PDF_PATH}'. Please re-index the document first."
    #},
    {
        "id": "csv_query_sales",
        "description": "Querying the sample CSV for total sales by department.",
        "text": f"Using the data in '{SAMPLE_CSV_PATH}', what are the total sales for each department?"
    },
    #{
    #    "id": "csv_query_high_earners",
    #    "description": "Querying the sample CSV for high earners.",
    #    "text": f"From the file '{SAMPLE_CSV_PATH}', list employees with sales greater than 150."
    #},
    {
        "id": "web_search_query",
        "description": "Performing a web search.",
        "text": "What are the latest advancements in AI agent technology?"
    }
]

async def run_agent_demo():
    logger.info("Starting PharmaDB Research Agent Demo")

    # --- Configuration Check ---
    default_llm_settings = config.llm.get("default")

    if not default_llm_settings or not default_llm_settings.api_key:
        logger.error("Default LLM configuration (e.g., [llm.default]) or API key is not set in openmanus_core/config/config.toml. Aborting demo.")
        logger.error("Please ensure your [llm.default] section has 'model', 'base_url', and 'api_key' set.")
        return

    logger.info(f"Using LLM: {default_llm_settings.model} via {default_llm_settings.base_url}")
    if default_llm_settings.embedding_model:
        logger.info(f"Using Embedding Model: {default_llm_settings.embedding_model}")

    # --- File Checks ---
    if not os.path.exists(SAMPLE_CSV_PATH):
        logger.error(f"Sample CSV file not found: {SAMPLE_CSV_PATH}. Please create it. Aborting CSV tests.")
        global PROMPTS
        PROMPTS = [p for p in PROMPTS if "csv_query" not in p["id"]]

    if not os.path.exists(SAMPLE_PDF_PATH):
        logger.error(f"Sample PDF file not found: {SAMPLE_PDF_PATH}. Please create it. Aborting PDF tests.")
        PROMPTS = [p for p in PROMPTS if "pdf_query" not in p["id"]]

    if not PROMPTS:
        logger.warning("No prompts to run after file checks. Exiting.")
        return

    # --- Initialize Agent ---
    logger.info("Initializing Manus agent...")
    try:
        # Ensure workspace_root is set for tools like PDFSearchTool
        # The Manus.create() or Manus() might handle its own config access.
        # If PDFSearchTool relies on a global config.workspace_root, it needs to be set.
        if not config.workspace_root:
            # Defaulting to openmanus_core/workspace as a common case for tools
            config.workspace_root = os.path.join(openmanus_core_path, "workspace") 
            logger.info(f"Set config.workspace_root to: {config.workspace_root}")
        
        agent = await Manus.create()  # Using the async factory method
        logger.info("Manus agent initialized.")
    except Exception as e:
        logger.error(f"Error initializing Manus agent: {e}", exc_info=True)
        return

    # --- Run Prompts ---
    for prompt_info in PROMPTS:
        logger.info(f"--- Running Prompt: {prompt_info['description']} ---")
        logger.info(f"Prompt Text: {prompt_info['text']}")
        try:
            # Clear agent's memory before each run for isolated tests
            if hasattr(agent, 'memory') and hasattr(agent.memory, 'clear'):
                agent.memory.clear()
                logger.info("Agent memory cleared for new prompt.")
            
            response = await agent.run(prompt_info['text'])
            logger.info(f"Agent's Final Response for '{prompt_info['id']}':\n{response}")
        except Exception as e:
            logger.error(f"Error running prompt '{prompt_info['id']}': {e}", exc_info=True)
        logger.info("--- Prompt Complete ---\n")

    logger.info("PharmaDB Research Agent Demo Finished")
    # Consider adding agent.cleanup() if Manus.create() has a corresponding cleanup method
    if hasattr(agent, 'cleanup') and asyncio.iscoroutinefunction(agent.cleanup):
        logger.info("Cleaning up agent resources...")
        await agent.cleanup()
        logger.info("Agent resources cleaned up.")

if __name__ == "__main__":
    # To run this demo: Ensure you are in the root directory of your project (`PharmaDB-research-agent`)
    # and execute: python demo_agent.py
    # Make sure your openmanus_core/config/config.toml is correctly set up with API keys.
    # And that sample_data.csv and sample_doc.pdf are in openmanus_core/workspace/
    try:
        asyncio.run(run_agent_demo())
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred at the top level: {e}", exc_info=True) 