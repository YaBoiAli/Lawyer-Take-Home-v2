import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from modules.ocr_module import DocumentOCR
from modules.neon_db_module import get_db_connection, create_documents_table, insert_ocr_data, fuzzy_search_all_documents
from modules.rag_module import Neo4jRAGSystem
from dotenv import load_dotenv
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
print(f"DEBUG: NEO4J_URI loaded from environment: {NEO4J_URI}")

# Instantiate your modules so they are ready to use
# This is more efficient than creating them on each request
print("Initializing services...")
# OCR_PROCESSOR will be initialized dynamically based on the chosen provider
RAG_SYSTEM = Neo4jRAGSystem(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, GROQ_API_KEY) # RAG still uses Groq for now
print("Services initialized.")

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
    and ingest into both databases.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not file or not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed types are: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file_extension = filename.rsplit('.', 1)[1].lower()
    
    extracted_data = []

    # Get LLM provider from form data, default to 'groq'
    llm_provider = request.form.get('llm_provider', 'groq').lower()
    
    # Initialize OCR_PROCESSOR dynamically based on the chosen provider
    if llm_provider == "groq":
        ocr_processor = DocumentOCR(groq_api_key=GROQ_API_KEY, llm_provider="groq")
    elif llm_provider == "openrouter":
        ocr_processor = DocumentOCR(openrouter_api_key=OPENROUTER_API_KEY, llm_provider="openrouter")
    else:
        return jsonify({"error": f"Unsupported LLM provider: {llm_provider}. Choose 'groq' or 'openrouter'."}), 400
    
    try:
        file.save(filepath)

        if file_extension in {'pdf', 'png', 'jpg', 'jpeg', 'gif'}:
            print(f"Performing OCR on {filename} using {llm_provider}...")
            if file_extension == 'pdf':
                extracted_data = ocr_processor.process_pdf_pages(filepath)
            else: # Image file
                page_data = ocr_processor.extract_text_from_image(filepath)
                page_data['page_number'] = 1 # For images, consider it page 1
                page_data['file_path'] = os.path.basename(filepath)
                extracted_data.append(page_data)

            if not extracted_data:
                return jsonify({"error": "OCR failed to extract any text."}), 500
            print(f"OCR complete. Extracted {len(extracted_data)} pages/images.")

        elif file_extension in {'xlsx', 'xls', 'csv'}:
            print(f"Processing spreadsheet {filename}...")
            spreadsheet_processor = SpreadsheetProcessor()
            extracted_data = spreadsheet_processor.process_spreadsheet(filepath)
            if not extracted_data:
                return jsonify({"error": "Spreadsheet processing failed to extract any data."}), 500
            print(f"Spreadsheet processing complete. Extracted {len(extracted_data)} rows/sheets.")
        else:
            return jsonify({"error": "Unsupported file type for extraction."}), 400

        # 2. Ingest into Neon DB
        print("Ingesting data into Neon DB...")
        with get_db_connection(NEON_CONNECTION_STRING) as neon_conn:
            if neon_conn:
                create_documents_table(neon_conn)
                insert_ocr_data(neon_conn, extracted_data) # This function can handle the format
            else:
                print("Warning: Could not connect to Neon DB.")
        
        # 3. Ingest into Neo4j
        print("Ingesting data into Neo4j...")
        if RAG_SYSTEM.driver:
            RAG_SYSTEM.ingest_data(extracted_data)
        else:
            print("Warning: RAG system not connected.")

        return jsonify({
            "message": f"Successfully processed and ingested '{filename}'.",
            "items_processed": len(extracted_data)
        }), 200

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up the uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/api/fuzzy-search', methods=['POST'])
def fuzzy_search():
    """
    Endpoint to perform a fuzzy search across all documents in Neon DB.
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Query is required"}), 400
    
    query = data['query']
    
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
            return jsonify(formatted_results), 200
            
    except Exception as e:
        print(f"An error occurred during fuzzy search: {e}")
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

    try:
        if not RAG_SYSTEM.driver:
            return jsonify({"error": "RAG system is not available"}), 500
            
        answer = RAG_SYSTEM.generate_answer(query, file_name=file_name)
        return jsonify({"answer": answer}), 200

    except Exception as e:
        print(f"An error occurred during RAG search: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5001)
