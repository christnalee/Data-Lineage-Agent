# Data-Lineage-Agent

This repository hosts an LLM-Powered Knowledge Agent designed to bring visibility and understanding to a data ecosystem. By extracting, synthesizing, and presenting information from system logs, this agent empowers teams to navigate complex data lineage, transformations, and dependencies with ease.

### The Challenge: Lack of Visibility

The data ecosystem generates a vast amount of data through complex processing steps. This abundance of information previously led to significant challenges:

- Limited Insight: Quickly understanding critical aspects like data lineage (where data originates), data transformations (how it's processed), and component dependencies was a painstaking process.
- Debugging Difficulties: The lack of clear visibility made troubleshooting issues time-consuming and frustrating.
- Duplicated Efforts: Unclear ownership and the absence of a centralized knowledge source led to duplicated efforts and missing existing solutions.
- Slowdowns in Innovation: Valuable time was spent deciphering existing data rather than driving new product development and innovation.

### Solution: LLM-Powered Knowledge Agent

To overcome these challenges, I've developed an intelligent system that leverages the power of Large Language Models (LLMs) and the Retrieval-Augmented Generation (RAG) paradigm. This agent:

- Extracts & Synthesizes: Derives insights directly from system logs.
- Natural Language Understanding: Utilizes a local Llama 3 model to comprehend natural language queries.
- Clear Answers: Provides concise and accurate responses regarding data lineage, transformations, and usage.
- Enhanced Query Flexibility: Understands a wider range of prompts using semantic embeddings, enabling more intuitive interaction.
- Embedding Visualization: Offers tools to visually explore the semantic relationships within our data and queries.

### How it Works: The RAG Paradigm

The core of our agent operates on the Retrieval-Augmented Generation (RAG) principle. This means it doesn't rely solely on the LLM's pre-trained knowledge. Instead, it augments the LLM's capabilities by first retrieving relevant information from our structured log data.

- Retrieval: When a query is received, the agent uses ChromaDB (our vector database) to find semantically similar data points from the parsed logs. This is achieved by converting both the query and the log data into vector embeddings using Sentence Transformers.
- Augmentation: The retrieved relevant context (e.g., specific table lineage, transformation logic) is then combined with the original user query.
- Generation: This combined information is fed into our local Llama 3 LLM. The LLM then generates an answer that is grounded in the factual data retrieved from the logs, ensuring accuracy and minimizing hallucinations.

### Key Features & Capabilities

Our agent is built to provide comprehensive data insights:

- Comprehensive Data Discovery: List and understand all known tables from both Hive and Spark, along with their properties.
- Detailed Lineage Tracing: Map data flow from source to output, providing a clear visual of data journeys.
- Transformation Logic Insight: Explain how data is transformed, detailing source columns and aggregation types.
- Cross-System Visibility: Offer unified insights into both Hive and Spark data pipelines, bridging the gap between them.
- Embedding Visualization: Gain a deeper, visual understanding of how different data points and your own queries relate to each other.

### Core Components & Process:

### 1. Data Ingestion & Parsing:

- Raw Data Sources: Hive Logs (SQL query structures, data vertices) and Spark Logs (job execution details, input/output datasets, column info, lineage).
- Parsers: Custom Python scripts (Hive Parser, Spark Parser) extract detailed, structured information.
- Output: Processed data is stored in structured text files, forming our knowledge base.

### 2. Initialization:

- Data Loading: Specialized loaders read the processed Hive and Spark data.
- Schema Extraction: A unified schema is built from both sources, creating a factual base of tables, columns, and relationships.
- Vector Store Setup: Parsed data entries are loaded into ChromaDB, our vector database, using Sentence Transformer embeddings with associated metadata for richer retrieval.
- LLM Service Initialization: The local Llama 3 model is loaded and made ready for inference.

### 3. Retrieval Augmented Generation (RAG):

- Intent Classification & Pre-processing: User queries are refined using embedding understanding.
- Factual Data Retrieval & Processing: ChromaDB is queried using Sentence Transformer embeddings to retrieve semantically similar context from our knowledge base.
- Intelligent Prompt Construction: Retrieved context and the refined query are used to construct a prompt for the LLM.
- LLM Reasoning & Answer Generation: The local Llama 3 model generates answers, grounded in the retrieved factual data.
- Post-processing: Outputs are refined for clarity and accuracy.
- Embedding Visualization: The same ChromaDB data is used to power visualization tools, allowing users to explore semantic relationships between data and queries.

### How to Run

To get started with the Knowledge Agent:

### 1. Set up your Environment:

- Clone this repository.
- Create a virtual environment (recommended):
    - python -m venv venv
    - source venv/bin/activate # On Windows use `venv\\Scripts\\activate`
- Install all necessary dependencies:
    - pip install -r requirements.txt

### 2. Place Logs:

- Put the Hive logs you want to parse into the ./hive_logs folder.
- Put the Spark logs you want to parse into the ./spark_logs folder.

### 3. Parse Logs:

- Run the Hive log parser:
    - python parse_hive.py
    - This will generate extracted_hive.txt.
- Run the Spark log parser:
    - python parse_spark.py
    - This will generate extracted_spark.txt.

### 4. Run the Main Agent:

- Start the interactive agent by running:
    - python [main.py](http://main.py/)

### Technical Stack & Tools

- Programming Language: Python
- Data Processing & Analysis: Custom Python scripts for log parsing and schema extraction.
- Vector Database: ChromaDB for efficient similarity search.
- Embeddings: Sentence Transformers (all-MiniLM-L6-v2) for converting data into vector representations.
- LLM: Local Llama 3 8B Instruct (Meta-Llama-3-8B-Instruct.Q4_K_M.gguf) for intelligent response generation.
- SQL Parsing: sqlparse library for understanding Hive query structures.
- Data Storage: Local file system for processed data.
- Configuration: dotenv for environment variables and dataclasses for application settings.

### Query Examples

### Hive

- List all tables.
- What is the downstream of the first one?
- What is the upstream of [table name]?
- What is the upstream and downstream of [table name]?
- What are the known table connections?
- List all the columns in [table name].
- How is column [column name] populated in table [table name]?
- How is [column name] populated in [table name]?

### Spark

- What is the upstream of [table name]?
- What are the columns and their data types for [table name]?
- How is the [column name] column in the output derived?
- What are all the source columns that contribute to the [column name] column?
- Is [table name] used in any aggregations for the output table? If so, for which column?

### Value Proposition

- Increased Efficiency: Reduce time spent on data exploration and understanding.
- Improved Collaboration: Foster a shared understanding of data across teams.
- Accelerated Innovation: Enable faster development cycles for data-intensive products.
- Enhanced Data Governance: Contribute to better understanding and management of data assets.
- Detailed Visualization: Provide clear, visual insights into Q&A embeddings.

### Next Steps

- Expand Knowledge Base: Integrate more log sources and historical data for broader coverage.
- Enhance Parsing: Improve robustness and the detail extracted by our parsers.
- User-Friendly Web Interface: Develop an intuitive interface for easier access to querying and visualization features, including making the embeddings visualization more interactive.
- Advanced Features: Support for more complex query types, entity linking, and proactive alerts, potentially leveraging insights from embedding analysis.
- Feedback Loop: Incorporate user feedback to continuously refine the agent’s performance and the usefulness of its visualizations.