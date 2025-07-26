# OCR and GraphRAG Project with Structured Data Processing

This project combines Optical Character Recognition (OCR) with a Graph-based Retrieval Augmented Generation (RAG) system, now enhanced with **structured data processing** capabilities for better document understanding and entity-based querying.

## 🆕 New Features: Structured Data Processing

### Enhanced Document Processing
- **Entity Extraction**: Automatically identifies people, organizations, dates, phone numbers, emails, and more
- **Table Detection**: Converts table-like text into properly formatted markdown tables
- **Signature Recognition**: Identifies and formats signature blocks and signatory information
- **Markdown Formatting**: Creates well-structured markdown content with headers, tables, and entity summaries
- **Enhanced Searchability**: Generates searchable terms including name variations and related keywords

### Entity-Based Querying
Now you can make queries like:
- "Find all info on Miguel A Vasquez"
- "Show me documents mentioning ACME Corporation"
- "What contracts were signed in March 2024?"
- "Find all invoices with payment terms"

## How to Run the Project
### Without Docker

To run this project without Docker, follow these steps:

1.  **Create and Activate Virtual Environment**:
    It's recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

2.  **Install Dependencies**:
    With your virtual environment activated, install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install spaCy Language Model**:
    For optimal entity extraction, install the English language model:
    ```bash
    python -m spacy download en_core_web_sm
    ```

4.  **Set up Environment Variables**:
    Create a `.env` file in the root directory of the project. This file should contain the following environment variables:
    *   `NEON_CONNECTION_STRING`: Your Neon database connection URL.
    *   `GROQ_API_KEY`: Your Groq API key for LLM processing.
    *   `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`: Neo4j database credentials.
    
    Example `.env` content:
    ```
    GROQ_API_KEY=your_groq_api_key_here
    NEON_CONNECTION_STRING="postgresql://user:password@host/database"
    NEO4J_URI="bolt://localhost:7687"
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=your_neo4j_password
    ```

5.  **Test Structured Processing** (Optional):
    Test the new structured data capabilities:
    ```bash
    python test_structured_processing.py
    ```

6.  **Run the Application**:
    Start the Flask application:
    ```bash
    python app.py
    ```
    The application should now be running, typically accessible at `http://127.0.0.1:5001`.

### With Docker

To run this project using Docker and Docker Compose, follow these steps:

1.  **Install Docker**:
    Ensure you have Docker and Docker Compose installed on your system. You can download them from the official Docker website.

2.  **Set up Environment Variables**:
    Create a `.env` file in the root directory of the project and add necessary environment variables as described in the "Without Docker" section above. These variables will be picked up by Docker Compose.

3.  **Build and Run with Docker Compose**:
    Navigate to the root directory of the project in your terminal and run:
    ```bash
    docker-compose up --build
    ```
    This command will build the Docker image (if it hasn't been built yet or if changes were made to the `Dockerfile`) and start the services defined in `docker-compose.yml`. The application should then be accessible, typically at `http://localhost:5001`.

4.  **Stop the Application**:
    To stop the running Docker containers, press `Ctrl+C` in the terminal where `docker-compose up` is running. To remove the containers, networks, and volumes created by `up`, run:
    ```bash
    docker-compose down
    docker-compose down -v      # Deletes everything saved locally via volumes
    ```

## API Endpoints

### Original Endpoints
- `POST /api/upload-and-extract-data`: Upload and process documents
- `POST /api/fuzzy-search`: Fuzzy text search
- `POST /api/rag-search`: Basic RAG search

### 🆕 New Structured Data Endpoints
- `POST /api/structured-rag-search`: Enhanced RAG with markdown formatting and entity awareness
- `POST /api/entity-search`: Search for documents containing specific entities
- `GET /api/entity-info/<entity_name>`: Get comprehensive information about an entity
- `POST /api/advanced-search`: Combined text and entity search
- `GET /api/entity-statistics`: Get statistics about entities in the database

### Example API Usage

#### Entity Search
```bash
curl -X POST http://localhost:5001/api/entity-search \
  -H "Content-Type: application/json" \
  -d '{
    "entity_name": "Miguel A Vasquez",
    "entity_type": "PERSON",
    "limit": 10
  }'
```

#### Structured RAG Search
```bash
curl -X POST http://localhost:5001/api/structured-rag-search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the salary information?",
    "entity_filter": "Miguel A Vasquez"
  }'
```

#### Advanced Search
```bash
curl -X POST http://localhost:5001/api/advanced-search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "employment contract",
    "entity_filters": ["Miguel A Vasquez", "ACME Corporation"],
    "limit": 20
  }'
```

## Project Modules

The project is structured into several modules, each handling specific functionalities:

*   `app.py`: The main Flask application file, handling routes and overall application logic.
*   `modules/ocr_module.py`: Contains logic for Optical Character Recognition.
*   `modules/rag_module.py`: Implements the Retrieval Augmented Generation logic with structured data support.
*   `modules/neon_db_module.py`: Manages interactions with the Neon database, including structured data storage.
*   🆕 `modules/structured_data_processor.py`: **New module** for converting OCR text into structured formats with entity extraction, table detection, and markdown formatting.
*   `modules/spreadsheet_processor.py`: Handles spreadsheet file processing.
*   `static/`: Contains static files like `script.js` (JavaScript) and `styles.css` (CSS).
*   `templates/`: Stores HTML templates, such as `index.html`.

## Database Schema

### Original Tables
- `documents`: Stores raw OCR text with basic metadata
- Neo4j `Chunk` nodes: Basic text chunks with embeddings

### 🆕 Enhanced Schema
- `structured_documents`: Stores processed documents with entities, markdown content, and metadata
- Neo4j `StructuredChunk` nodes: Enhanced chunks with markdown content and entity relationships
- Neo4j `Entity` nodes: Extracted entities (people, organizations, etc.) with types and confidence scores

## Structured Data Processing Pipeline

1. **OCR/Text Extraction**: Extract raw text from documents
2. **Entity Extraction**: Identify people, organizations, dates, contacts using spaCy NLP + custom patterns
3. **Table Detection**: Recognize table-like structures and convert to markdown format
4. **Signature Identification**: Find and format signature blocks
5. **Markdown Generation**: Create structured markdown with headers, tables, and entity summaries
6. **Searchability Enhancement**: Generate searchable terms including name variations
7. **Database Storage**: Store in both PostgreSQL (structured_documents) and Neo4j (StructuredChunk + Entity nodes)

## System Architecture

The following diagram illustrates the high-level architecture and data flow of the project:

![System Architecture](architecture-diagram.png)

**Description:**
- **User Interaction:** Users interact with the system via the web interface to upload files, perform fuzzy or RAG search queries, and view results.
- **Frontend:** The frontend (`index.html` and `script.js`) handles user input and communicates with the backend through API endpoints.
- **Backend:** The backend (`app.py`) processes requests, coordinates between modules, and returns results to the frontend.
- **🆕 Structured Processing:** The new `structured_data_processor.py` converts raw OCR text into structured format with entities, tables, and markdown.
- **Data Processing Modules:**
  - `ocr_module.py` handles OCR and file type detection for PDFs and images.
  - `spreadsheet_processor.py` processes spreadsheet files.
- **Database Modules:**
  - `neon_db_module.py` and `rag_module.py` manage data ingestion, querying, and results retrieval from the Neon and Neo4j databases, respectively.
- **Databases:** Data is stored and retrieved from Neon and Neo4j databases with enhanced structured data support.

## Example Use Cases

### Legal Document Processing
- **Contract Analysis**: Extract parties, dates, terms, signatures
- **Invoice Processing**: Identify billing entities, amounts, payment terms
- **Agreement Review**: Find specific clauses, signatory information

### Business Document Management  
- **Employee Records**: Track personnel information across documents
- **Vendor Management**: Consolidate supplier information and contracts
- **Compliance Tracking**: Monitor document signatures and dates

### Research and Discovery
- **Entity Profiling**: Get comprehensive view of any person or organization
- **Timeline Analysis**: Track events and dates across document sets
- **Relationship Mapping**: Understand connections between entities

## Performance Features

- **Duplicate Prevention**: Avoids re-processing existing documents
- **Batch Processing**: Efficiently handles multiple documents
- **Indexed Search**: Fast entity and text searches using PostgreSQL GIN indexes
- **Vector Similarity**: Semantic search using sentence transformers
- **Graph Relationships**: Complex entity relationship queries via Neo4j

> **Note:** The diagram file should be named `architecture-diagram.png` and placed in the project root. If you want to use the provided diagram, save it as `architecture-diagram.png` in the root directory.
