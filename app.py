import os
import logging
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from modules.ocr_module import DocumentOCR
from modules.neon_db_module import (
    get_db_connection, create_documents_table, insert_ocr_data, fuzzy_search_all_documents,
    create_structured_documents_table, insert_structured_data, search_by_entity, 
    get_entity_info, advanced_entity_search, get_entity_statistics
)
from modules.rag_module import Neo4jRAGSystem
from modules.structured_data_processor import StructuredDataProcessor
from dotenv import load_dotenv
import json

# Set up comprehensive logging for markdown processing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('markdown_processing.log'),
        logging.StreamHandler()
    ]
)
app_logger = logging.getLogger('MarkdownApp')

load_dotenv()
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'xlsx', 'xls', 'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit for uploads

# Load credentials once when the app starts
# Make sure to set these as environment variables for production
# For Colab/local, you can define them here or use a .env file
GROQ_API_KEY = os.environ.get('GROQ_API_KEY') # Replace with userdata.get() in Colab
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
NEON_CONNECTION_STRING = os.environ.get('NEON_CONNECTION_STRING')
NEO4J_URI = os.environ.get('NEO4J_URI')
NEO4J_USER = os.environ.get('NEO4J_USER')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')

# Debugging: Print the loaded Neo4j URI to verify
app_logger.info(f"🔧 NEO4J_URI loaded from environment: {NEO4J_URI}")

# Instantiate your modules so they are ready to use
# This is more efficient than creating them on each request
app_logger.info("🚀 Initializing services...")
# OCR_PROCESSOR will be initialized dynamically based on the chosen provider
RAG_SYSTEM = Neo4jRAGSystem(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, GROQ_API_KEY) # RAG still uses Groq for now
STRUCTURED_PROCESSOR = StructuredDataProcessor(GROQ_API_KEY)  # Initialize structured data processor
app_logger.info("✅ Services initialized successfully")

from modules.spreadsheet_processor import SpreadsheetProcessor

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- FRONTEND ROUTE ---
@app.route('/')
def index():
    """Renders the main frontend page."""
    return render_template('index.html')

# --- API ENDPOINTS ---

@app.route('/api/upload-and-process', methods=['POST'])
def upload_and_process_file():
    """
    Endpoint to upload a PDF, perform OCR, and ingest into both databases.
    This endpoint is deprecated in favor of /api/upload-and-extract-data.
    """
    return jsonify({"error": "This endpoint is deprecated. Please use /api/upload-and-extract-data instead."}), 400

@app.route('/api/upload-and-extract-data', methods=['POST'])
def upload_and_extract_data():
    """
    Endpoint to upload a file (PDF, image, or spreadsheet), extract data,
    and ingest into both databases with structured processing.
    """
    app_logger.info("📤 New file upload request received")
    
    if 'file' not in request.files:
        app_logger.warning("❌ No file part in request")
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        app_logger.warning("❌ No file selected")
        return jsonify({"error": "No selected file"}), 400
    if not file or not allowed_file(file.filename):
        app_logger.warning(f"❌ Invalid file type: {file.filename}")
        return jsonify({"error": f"Invalid file type. Allowed types are: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file_extension = filename.rsplit('.', 1)[1].lower()
    
    app_logger.info(f"📁 Processing file: {filename} (type: {file_extension})")
    
    extracted_data = []

    # Get LLM provider from form data, default to 'groq'
    llm_provider = request.form.get('llm_provider', 'groq').lower()
    app_logger.info(f"🤖 Using LLM provider: {llm_provider}")
    
    # Initialize OCR_PROCESSOR dynamically based on the chosen provider
    if llm_provider == "groq":
        ocr_processor = DocumentOCR(groq_api_key=GROQ_API_KEY, llm_provider="groq")
    elif llm_provider == "openrouter":
        ocr_processor = DocumentOCR(openrouter_api_key=OPENROUTER_API_KEY, llm_provider="openrouter")
    else:
        app_logger.error(f"❌ Unsupported LLM provider: {llm_provider}")
        return jsonify({"error": f"Unsupported LLM provider: {llm_provider}. Choose 'groq' or 'openrouter'."}), 400
    
    try:
        file.save(filepath)
        app_logger.info(f"💾 File saved to: {filepath}")

        if file_extension in {'pdf', 'png', 'jpg', 'jpeg', 'gif'}:
            app_logger.info(f"🔍 Performing OCR on {filename} using {llm_provider}...")
            if file_extension == 'pdf':
                extracted_data = ocr_processor.process_pdf_pages(filepath)
            else: # Image file
                page_data = ocr_processor.extract_text_from_image(filepath)
                page_data['page_number'] = 1 # For images, consider it page 1
                page_data['file_path'] = os.path.basename(filepath)
                extracted_data.append(page_data)

            if not extracted_data:
                app_logger.error("❌ OCR failed to extract any text")
                return jsonify({"error": "OCR failed to extract any text."}), 500
            app_logger.info(f"✅ OCR complete. Extracted {len(extracted_data)} pages/images.")

        elif file_extension in {'xlsx', 'xls', 'csv'}:
            app_logger.info(f"📊 Processing spreadsheet {filename}...")
            spreadsheet_processor = SpreadsheetProcessor()
            extracted_data = spreadsheet_processor.process_spreadsheet(filepath)
            if not extracted_data:
                app_logger.error("❌ Spreadsheet processing failed")
                return jsonify({"error": "Spreadsheet processing failed to extract any data."}), 500
            app_logger.info(f"✅ Spreadsheet processing complete. Extracted {len(extracted_data)} rows/sheets.")
        else:
            app_logger.error(f"❌ Unsupported file type for extraction: {file_extension}")
            return jsonify({"error": "Unsupported file type for extraction."}), 400

        # 2. Process data into structured format
        app_logger.info("🏗️ Processing data into structured markdown format...")
        structured_docs = STRUCTURED_PROCESSOR.process_batch(extracted_data)
        app_logger.info(f"✅ Structured processing complete. Created {len(structured_docs)} structured documents.")
        
        # Log markdown statistics
        total_markdown_length = sum(len(doc.markdown_content) for doc in structured_docs)
        app_logger.info(f"📝 Total markdown content generated: {total_markdown_length} characters")
        
        # 3. Ingest into Neon DB (both original and structured)
        app_logger.info("💾 Ingesting data into Neon DB...")
        with get_db_connection(NEON_CONNECTION_STRING) as neon_conn:
            if neon_conn:
                app_logger.info("🔗 Connected to Neon DB successfully")
                # Create both tables
                create_documents_table(neon_conn)
                create_structured_documents_table(neon_conn)
                
                # Insert original OCR data
                insert_ocr_data(neon_conn, extracted_data)
                
                # Insert structured data
                insert_structured_data(neon_conn, structured_docs)
                app_logger.info("✅ Data successfully ingested into Neon DB")
            else:
                app_logger.warning("⚠️ Could not connect to Neon DB")
        
        # 4. Ingest into Neo4j (both original and structured)
        app_logger.info("🕸️ Ingesting data into Neo4j...")
        if RAG_SYSTEM.driver:
            app_logger.info("🔗 Connected to Neo4j successfully")
            # Ingest original data
            RAG_SYSTEM.ingest_data(extracted_data)
            
            # Ingest structured data
            RAG_SYSTEM.ingest_structured_data(structured_docs)
            app_logger.info("✅ Data successfully ingested into Neo4j")
        else:
            app_logger.warning("⚠️ RAG system not connected")

        # Calculate statistics for response
        total_entities = sum(len(doc.entities) for doc in structured_docs)
        total_tables = sum(len(doc.tables) for doc in structured_docs)
        total_signatures = sum(len(doc.signatures) for doc in structured_docs)
        
        app_logger.info("🎉 File processing complete!")
        app_logger.info(f"📊 Final statistics:")
        app_logger.info(f"   📄 Documents processed: {len(extracted_data)}")
        app_logger.info(f"   🏗️ Structured documents: {len(structured_docs)}")
        app_logger.info(f"   👤 Entities extracted: {total_entities}")
        app_logger.info(f"   📋 Tables detected: {total_tables}")
        app_logger.info(f"   ✍️ Signatures found: {total_signatures}")
        app_logger.info(f"   📝 Markdown generated: {total_markdown_length} chars")

        return jsonify({
            "message": f"Successfully processed and ingested '{filename}' with structured data.",
            "items_processed": len(extracted_data),
            "structured_documents": len(structured_docs),
            "entities_extracted": total_entities,
            "tables_detected": total_tables,
            "signatures_found": total_signatures,
            "markdown_length": total_markdown_length
        }), 200

    except Exception as e:
        app_logger.error(f"💥 An error occurred during processing: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up the uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)
            app_logger.info(f"🗑️ Cleaned up temporary file: {filepath}")


@app.route('/api/fuzzy-search', methods=['POST'])
def fuzzy_search():
    """
    Endpoint to perform a fuzzy search across all documents in Neon DB.
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Query is required"}), 400
    
    query = data['query']
    app_logger.info(f"🔍 Fuzzy search request: '{query}'")
    
    try:
        with get_db_connection(NEON_CONNECTION_STRING) as neon_conn:
            if not neon_conn:
                return jsonify({"error": "Could not connect to the database"}), 500
            
            results = fuzzy_search_all_documents(neon_conn, query)
            
            # Format results to be JSON-friendly
            formatted_results = [
                {
                    "file_name": row[0],
                    "page_number": row[1],
                    "text_preview": row[2][:200] + "...", # Send a preview
                    "similarity_score": round(row[3], 3)
                } for row in results
            ]
            
            app_logger.info(f"✅ Fuzzy search complete: {len(formatted_results)} results found")
            return jsonify(formatted_results), 200
            
    except Exception as e:
        app_logger.error(f"❌ Error during fuzzy search: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/rag-search', methods=['POST'])
def rag_search():
    """
    Endpoint to perform a RAG search using the Neo4j system.
    Accepts an optional 'file_name' to filter the search.
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Query is required"}), 400
        
    query = data['query']
    file_name = data.get('file_name') # Get optional file_name
    
    app_logger.info(f"🤖 RAG search request: '{query}'" + (f" (file: {file_name})" if file_name else ""))

    try:
        if not RAG_SYSTEM.driver:
            return jsonify({"error": "RAG system is not available"}), 500
            
        answer = RAG_SYSTEM.generate_answer(query, file_name=file_name)
        app_logger.info(f"✅ RAG search complete: {len(answer)} character response")
        return jsonify({"answer": answer}), 200

    except Exception as e:
        app_logger.error(f"❌ Error during RAG search: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/structured-rag-search', methods=['POST'])
def structured_rag_search():
    """
    Enhanced RAG search endpoint that uses structured data with markdown formatting
    and entity-aware retrieval. Supports entity filtering for specific person/organization queries.
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Query is required"}), 400
        
    query = data['query']
    file_name = data.get('file_name')  # Optional file filter
    entity_filter = data.get('entity_filter')  # Optional entity filter (e.g., "Miguel A Vasquez")
    
    app_logger.info(f"🎯 Structured RAG search: '{query}'" + 
                   (f" (file: {file_name})" if file_name else "") +
                   (f" (entity: {entity_filter})" if entity_filter else ""))

    try:
        if not RAG_SYSTEM.driver:
            return jsonify({"error": "RAG system is not available"}), 500
            
        answer = RAG_SYSTEM.generate_structured_answer(query, file_name=file_name, entity_filter=entity_filter)
        app_logger.info(f"✅ Structured RAG search complete: {len(answer)} character response with markdown")
        return jsonify({"answer": answer, "type": "structured"}), 200

    except Exception as e:
        app_logger.error(f"❌ Error during structured RAG search: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/entity-search', methods=['POST'])
def entity_search():
    """
    Endpoint to search for documents containing specific entities.
    Example: Find all documents mentioning "Miguel A Vasquez"
    """
    data = request.get_json()
    if not data or 'entity_name' not in data:
        return jsonify({"error": "Entity name is required"}), 400
    
    entity_name = data['entity_name']
    entity_type = data.get('entity_type')  # Optional: PERSON, ORG, etc.
    limit = data.get('limit', 10)
    
    app_logger.info(f"👤 Entity search: '{entity_name}'" + (f" (type: {entity_type})" if entity_type else ""))
    
    try:
        with get_db_connection(NEON_CONNECTION_STRING) as neon_conn:
            if not neon_conn:
                return jsonify({"error": "Could not connect to the database"}), 500
            
            results = search_by_entity(neon_conn, entity_name, entity_type, limit)
            
            # Format results
            formatted_results = []
            for row in results:
                file_name, page_number, markdown_content, entities_json, metadata = row
                formatted_results.append({
                    "file_name": file_name,
                    "page_number": page_number,
                    "markdown_preview": markdown_content[:300] + "..." if len(markdown_content) > 300 else markdown_content,
                    "entities": entities_json,
                    "metadata": metadata
                })
            
            app_logger.info(f"✅ Entity search complete: {len(formatted_results)} documents found")
            return jsonify({
                "entity_searched": entity_name,
                "results_count": len(formatted_results),
                "results": formatted_results
            }), 200
            
    except Exception as e:
        app_logger.error(f"❌ Error during entity search: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/entity-info/<entity_name>', methods=['GET'])
def get_entity_information(entity_name):
    """
    Endpoint to get comprehensive information about a specific entity.
    Returns all occurrences with context and metadata.
    """
    app_logger.info(f"📋 Entity info request: '{entity_name}'")
    
    try:
        with get_db_connection(NEON_CONNECTION_STRING) as neon_conn:
            if not neon_conn:
                return jsonify({"error": "Could not connect to the database"}), 500
            
            entity_info = get_entity_info(neon_conn, entity_name)
            app_logger.info(f"✅ Entity info complete: {entity_info['total_occurrences']} occurrences found")
            return jsonify(entity_info), 200
            
    except Exception as e:
        app_logger.error(f"❌ Error getting entity info: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/advanced-search', methods=['POST'])
def advanced_search():
    """
    Advanced search endpoint that combines text search with entity filtering.
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Query is required"}), 400
    
    query = data['query']
    entity_filters = data.get('entity_filters', [])  # List of entities to filter by
    limit = data.get('limit', 20)
    
    app_logger.info(f"🔍 Advanced search: '{query}' with entity filters: {entity_filters}")
    
    try:
        with get_db_connection(NEON_CONNECTION_STRING) as neon_conn:
            if not neon_conn:
                return jsonify({"error": "Could not connect to the database"}), 500
            
            results = advanced_entity_search(neon_conn, query, entity_filters, limit)
            
            # Format results
            formatted_results = []
            for row in results:
                file_name, page_number, markdown_content, entities_json, metadata, text_rank, similarity_score = row
                formatted_results.append({
                    "file_name": file_name,
                    "page_number": page_number,
                    "markdown_preview": markdown_content[:400] + "..." if len(markdown_content) > 400 else markdown_content,
                    "entities": entities_json,
                    "metadata": metadata,
                    "relevance_score": float(text_rank) if text_rank else 0,
                    "similarity_score": float(similarity_score) if similarity_score else 0
                })
            
            app_logger.info(f"✅ Advanced search complete: {len(formatted_results)} results found")
            return jsonify({
                "query": query,
                "entity_filters": entity_filters,
                "results_count": len(formatted_results),
                "results": formatted_results
            }), 200
            
    except Exception as e:
        app_logger.error(f"❌ Error during advanced search: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/entity-statistics', methods=['GET'])
def entity_statistics():
    """
    Endpoint to get statistics about entities in the database.
    """
    app_logger.info("📊 Entity statistics request")
    
    try:
        with get_db_connection(NEON_CONNECTION_STRING) as neon_conn:
            if not neon_conn:
                return jsonify({"error": "Could not connect to the database"}), 500
            
            stats = get_entity_statistics(neon_conn)
            app_logger.info(f"✅ Entity statistics complete: {stats['total_documents']} documents analyzed")
            return jsonify(stats), 200
            
    except Exception as e:
        app_logger.error(f"❌ Error getting entity statistics: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/debug/markdown/<file_name>', methods=['GET'])
def view_markdown_content(file_name):
    """
    Debug endpoint to view the raw markdown content of a processed document.
    Useful for verifying PDF to markdown conversion.
    """
    app_logger.info(f"🔍 Debug request for markdown content: {file_name}")
    
    try:
        with get_db_connection(NEON_CONNECTION_STRING) as neon_conn:
            if not neon_conn:
                return jsonify({"error": "Could not connect to the database"}), 500
            
            with neon_conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT file_name, page_number, markdown_content, metadata
                    FROM structured_documents 
                    WHERE file_name ILIKE %s
                    ORDER BY page_number
                    """,
                    (f'%{file_name}%',)
                )
                results = cur.fetchall()
                
                if not results:
                    return jsonify({"error": f"No markdown content found for file: {file_name}"}), 404
                
                # Format results for easy viewing
                markdown_pages = []
                for row in results:
                    file_name_db, page_number, markdown_content, metadata = row
                    
                    # Handle metadata safely - it could be string, dict, or None
                    try:
                        if isinstance(metadata, str):
                            parsed_metadata = json.loads(metadata)
                        elif isinstance(metadata, dict):
                            parsed_metadata = metadata
                        else:
                            parsed_metadata = {}
                    except (json.JSONDecodeError, TypeError):
                        parsed_metadata = {}
                    
                    markdown_pages.append({
                        "file_name": file_name_db,
                        "page_number": page_number,
                        "markdown_content": markdown_content,
                        "metadata": parsed_metadata,
                        "markdown_length": len(markdown_content) if markdown_content else 0,
                        "preview": markdown_content[:500] + "..." if markdown_content and len(markdown_content) > 500 else markdown_content or ""
                    })
                
                app_logger.info(f"✅ Found {len(markdown_pages)} pages of markdown content")
                return jsonify({
                    "file_searched": file_name,
                    "pages_found": len(markdown_pages),
                    "total_markdown_length": sum(len(page["markdown_content"]) for page in markdown_pages),
                    "pages": markdown_pages
                }), 200
                
    except Exception as e:
        app_logger.error(f"❌ Error retrieving markdown content: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/debug/markdown-stats', methods=['GET'])
def get_markdown_statistics():
    """
    Debug endpoint to get overall statistics about markdown processing.
    Shows how many documents have been converted and their characteristics.
    """
    app_logger.info("📊 Debug request for markdown statistics")
    
    try:
        with get_db_connection(NEON_CONNECTION_STRING) as neon_conn:
            if not neon_conn:
                return jsonify({"error": "Could not connect to the database"}), 500
            
            with neon_conn.cursor() as cur:
                # Get overall statistics
                cur.execute(
                    """
                    SELECT 
                        COUNT(*) as total_documents,
                        COUNT(DISTINCT file_name) as unique_files,
                        AVG(LENGTH(markdown_content)) as avg_markdown_length,
                        MAX(LENGTH(markdown_content)) as max_markdown_length,
                        SUM(LENGTH(markdown_content)) as total_markdown_length,
                        AVG((metadata->>'entity_count')::int) as avg_entities,
                        AVG((metadata->>'table_count')::int) as avg_tables,
                        AVG((metadata->>'signature_count')::int) as avg_signatures
                    FROM structured_documents 
                    WHERE markdown_content IS NOT NULL
                    """
                )
                stats = cur.fetchone()
                
                # Get recent conversions
                cur.execute(
                    """
                    SELECT file_name, page_number, 
                           LENGTH(markdown_content) as markdown_length,
                           metadata->>'entity_count' as entities,
                           metadata->>'table_count' as tables,
                           creation_timestamp
                    FROM structured_documents 
                    ORDER BY creation_timestamp DESC 
                    LIMIT 10
                    """
                )
                recent_conversions = cur.fetchall()
                
                app_logger.info(f"✅ Markdown statistics retrieved: {stats[0]} total documents")
                return jsonify({
                    "statistics": {
                        "total_documents": stats[0] or 0,
                        "unique_files": stats[1] or 0,
                        "avg_markdown_length": round(stats[2]) if stats[2] else 0,
                        "max_markdown_length": stats[3] or 0,
                        "total_markdown_generated": stats[4] or 0,
                        "avg_entities_per_doc": round(stats[5], 1) if stats[5] else 0,
                        "avg_tables_per_doc": round(stats[6], 1) if stats[6] else 0,
                        "avg_signatures_per_doc": round(stats[7], 1) if stats[7] else 0
                    },
                    "recent_conversions": [
                        {
                            "file_name": row[0],
                            "page_number": row[1],
                            "markdown_length": row[2],
                            "entities": row[3],
                            "tables": row[4],
                            "processed_at": row[5].isoformat() if row[5] else None
                        }
                        for row in recent_conversions
                    ]
                }), 200
                
    except Exception as e:
        app_logger.error(f"❌ Error getting markdown statistics: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app_logger.info("🚀 Starting LegalRAGv2 application with markdown processing...")
    app_logger.info(f"📁 Upload folder: {UPLOAD_FOLDER}")
    app_logger.info("🎯 Ready to process PDFs into structured markdown!")
    app_logger.info("🔍 Debug endpoints available:")
    app_logger.info("   GET /api/debug/markdown/<filename> - View markdown content")
    app_logger.info("   GET /api/debug/markdown-stats - View conversion statistics")
    app.run(debug=True, host='0.0.0.0', port=5001)
