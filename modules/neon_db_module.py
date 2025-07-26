
# --- Neon/PostgreSQL Database Module ---
# This module handles all low-level database operations for storing and retrieving OCR-processed document data.
# It uses psycopg2 to connect to a Neon (PostgreSQL) database and manages a 'documents' table optimized for text search.
#
# CHEAT SHEET:
# - get_db_connection: Connects to Neon DB
# - create_documents_table: Sets up table, extension, and index
# - insert_ocr_data: Inserts OCR results, avoids duplicates
# - fuzzy_search_all_documents: Fuzzy text search using pg_trgm
# - create_structured_documents_table: Sets up enhanced table for structured data
# - insert_structured_data: Inserts structured document data with entities
# - search_by_entity: Searches documents by entity (e.g., person names)
# - get_entity_info: Gets all information related to a specific entity
#
# Table: documents(file_id UUID, file_name, page_number, page_text, creation_timestamp)
# Table: structured_documents(file_id UUID, file_name, page_number, original_text, markdown_content, entities JSON, metadata JSON, searchable_terms TEXT[])
# Index: GIN on page_text for fast search
#
# ---
import psycopg2
import uuid
from datetime import datetime
import json
from typing import List, Dict, Any, Optional

# Establishes a connection to the Neon database using the provided connection string.
def get_db_connection(neon_connection_string):
    # Returns a psycopg2 connection object or None if connection fails.
    try:
        conn = psycopg2.connect(neon_connection_string)
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to Neon DB: {e}")
        return None

# Creates the 'documents' table, enables pg_trgm extension, and sets up a GIN index for fast text search.
def create_documents_table(conn):
    # Uses SQL to ensure the extension, table, and index exist.
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                file_id UUID PRIMARY KEY,
                file_name VARCHAR(255) NOT NULL,
                page_number VARCHAR(255) NOT NULL, 
                page_text TEXT,
                creation_timestamp TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_gin_page_text ON documents USING gin (page_text gin_trgm_ops);")
        conn.commit()
    print("Table 'documents', pg_trgm extension, and GIN index are ready.")

# Creates the enhanced 'structured_documents' table for storing processed structured data.
def create_structured_documents_table(conn):
    # Sets up table with JSON columns for entities/metadata and array for searchable terms.
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS structured_documents (
                file_id UUID PRIMARY KEY,
                file_name VARCHAR(255) NOT NULL,
                page_number INTEGER NOT NULL,
                original_text TEXT,
                markdown_content TEXT,
                entities JSONB,
                metadata JSONB,
                searchable_terms TEXT[],
                creation_timestamp TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        # Indexes for fast searching
        cur.execute("CREATE INDEX IF NOT EXISTS idx_gin_markdown ON structured_documents USING gin (markdown_content gin_trgm_ops);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_gin_entities ON structured_documents USING gin (entities);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_gin_searchable ON structured_documents USING gin (searchable_terms);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_file_name ON structured_documents (file_name);")
        conn.commit()
    print("Table 'structured_documents' and indexes are ready.")

# Inserts OCR data into the documents table, avoiding duplicates by checking (file_name, page_number) pairs.
def insert_ocr_data(conn, ocr_output):
    # ocr_output: List of dicts with keys 'file_path', 'page_number', 'full_text'.
    with conn.cursor() as cur:
        # Fetch all existing (file_name, page_number) to avoid duplicates.
        cur.execute("SELECT file_name, page_number FROM documents;")
        existing_records = {(row[0], row[1]) for row in cur.fetchall()}

        records_to_insert = []
        for page_data in ocr_output:
            file_name = page_data.get("file_path", "").split("/")[-1]
            page_number = page_data.get("page_number")
            page_text = page_data.get("full_text")

            # Only insert if this (file_name, page_number) is not already present.
            if (file_name, page_number) not in existing_records:
                records_to_insert.append((str(uuid.uuid4()), file_name, page_number, page_text))

        if records_to_insert:
            # Bulk insert new records.
            cur.executemany(
                "INSERT INTO documents (file_id, file_name, page_number, page_text) VALUES (%s, %s, %s, %s)",
                records_to_insert
            )
            conn.commit()
            print(f"{len(records_to_insert)} new records inserted successfully into Neon DB.")
        else:
            print("No new records to insert into Neon DB. Data may already exist.")

# Inserts structured document data into the enhanced table.
def insert_structured_data(conn, structured_docs):
    # structured_docs: List of StructuredDocument objects from structured_data_processor.
    with conn.cursor() as cur:
        # Check existing records to avoid duplicates
        cur.execute("SELECT file_name, page_number FROM structured_documents;")
        existing_records = {(row[0], row[1]) for row in cur.fetchall()}

        records_to_insert = []
        for doc in structured_docs:
            file_name = doc.metadata.get('file_name', '').split('/')[-1]
            page_number = doc.metadata.get('page_number', 1)
            
            # Only insert if this (file_name, page_number) is not already present.
            if (file_name, page_number) not in existing_records:
                # Convert entities to JSON format
                entities_json = [
                    {
                        'text': entity.text,
                        'label': entity.label,
                        'start': entity.start,
                        'end': entity.end,
                        'confidence': entity.confidence
                    }
                    for entity in doc.entities
                ]
                
                records_to_insert.append((
                    str(uuid.uuid4()),
                    file_name,
                    page_number,
                    doc.original_text,
                    doc.markdown_content,
                    json.dumps(entities_json),
                    json.dumps(doc.metadata),
                    doc.searchable_terms
                ))

        if records_to_insert:
            # Bulk insert structured records
            cur.executemany(
                """INSERT INTO structured_documents 
                   (file_id, file_name, page_number, original_text, markdown_content, entities, metadata, searchable_terms) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                records_to_insert
            )
            conn.commit()
            print(f"{len(records_to_insert)} structured records inserted successfully into Neon DB.")
        else:
            print("No new structured records to insert into Neon DB. Data may already exist.")

# Performs a case-insensitive fuzzy search across all documents using pg_trgm similarity.
def fuzzy_search_all_documents(conn, search_query, similarity_threshold=0.3):
    # Returns a list of tuples: (file_name, page_number, page_text, score)
    results = []
    with conn.cursor() as cur:
        # Set the similarity threshold for word_similarity function.
        cur.execute("SET pg_trgm.word_similarity_threshold = %s;", (similarity_threshold,))
        cur.execute(
            """
            SELECT file_name, page_number, page_text, word_similarity(LOWER(page_text), LOWER(%s)) AS score
            FROM documents WHERE LOWER(%s) <%% LOWER(page_text) ORDER BY score DESC;
            """,
            (search_query, search_query)
        )
        results = cur.fetchall()
    return results

# Searches structured documents by entity type and text (e.g., find all documents mentioning "Miguel A Vasquez").
def search_by_entity(conn, entity_text: str, entity_type: Optional[str] = None, limit: int = 10):
    # Returns documents containing the specified entity with structured information.
    results = []
    with conn.cursor() as cur:
        if entity_type:
            # Search for specific entity type and text
            cur.execute(
                """
                SELECT file_name, page_number, markdown_content, entities, metadata
                FROM structured_documents 
                WHERE entities @> %s
                ORDER BY creation_timestamp DESC
                LIMIT %s
                """,
                (json.dumps([{"text": entity_text, "label": entity_type}]), limit)
            )
        else:
            # Search for entity text regardless of type
            cur.execute(
                """
                SELECT file_name, page_number, markdown_content, entities, metadata
                FROM structured_documents 
                WHERE entities::text ILIKE %s
                ORDER BY creation_timestamp DESC
                LIMIT %s
                """,
                (f'%{entity_text}%', limit)
            )
        results = cur.fetchall()
    return results

# Gets comprehensive information about a specific entity across all documents.
def get_entity_info(conn, entity_text: str):
    # Returns all occurrences of an entity with context and metadata.
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 
                file_name, 
                page_number, 
                markdown_content,
                entities,
                metadata,
                searchable_terms
            FROM structured_documents 
            WHERE %s = ANY(searchable_terms) OR entities::text ILIKE %s
            ORDER BY file_name, page_number
            """,
            (entity_text.lower(), f'%{entity_text}%')
        )
        results = cur.fetchall()
        
        # Process results to extract relevant entity information
        entity_info = {
            'entity': entity_text,
            'total_occurrences': len(results),
            'documents': []
        }
        
        for row in results:
            file_name, page_number, markdown_content, entities_json, metadata, searchable_terms = row
            
            # Extract relevant entities from this document
            entities = json.loads(entities_json) if entities_json else []
            relevant_entities = [
                entity for entity in entities 
                if entity_text.lower() in entity['text'].lower()
            ]
            
            entity_info['documents'].append({
                'file_name': file_name,
                'page_number': page_number,
                'markdown_content': markdown_content,
                'relevant_entities': relevant_entities,
                'metadata': json.loads(metadata) if metadata else {},
                'context_preview': markdown_content[:300] + "..." if len(markdown_content) > 300 else markdown_content
            })
        
        return entity_info

# Advanced search that combines entity search with full-text search on structured content.
def advanced_entity_search(conn, query: str, entity_filters: Optional[List[str]] = None, limit: int = 20):
    # Performs sophisticated search combining multiple criteria.
    results = []
    with conn.cursor() as cur:
        base_query = """
        SELECT 
            file_name, 
            page_number, 
            markdown_content,
            entities,
            metadata,
            ts_rank(to_tsvector('english', markdown_content), plainto_tsquery('english', %s)) as text_rank,
            word_similarity(LOWER(markdown_content), LOWER(%s)) as similarity_score
        FROM structured_documents 
        WHERE 1=1
        """
        
        params = [query, query]
        
        # Add entity filters if provided
        if entity_filters:
            entity_conditions = []
            for entity in entity_filters:
                entity_conditions.append("entities::text ILIKE %s")
                params.append(f'%{entity}%')
            base_query += " AND (" + " OR ".join(entity_conditions) + ")"
        
        # Add text search condition
        base_query += " AND (to_tsvector('english', markdown_content) @@ plainto_tsquery('english', %s) OR %s = ANY(searchable_terms))"
        params.extend([query, query.lower()])
        
        # Order by relevance
        base_query += " ORDER BY text_rank DESC, similarity_score DESC LIMIT %s"
        params.append(limit)
        
        cur.execute(base_query, params)
        results = cur.fetchall()
    
    return results

# Gets statistics about entities across all documents.
def get_entity_statistics(conn):
    # Returns summary statistics about entities in the database.
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 
                COUNT(*) as total_documents,
                AVG(jsonb_array_length(entities)) as avg_entities_per_doc,
                COUNT(DISTINCT file_name) as unique_files
            FROM structured_documents 
            WHERE entities IS NOT NULL
            """
        )
        basic_stats = cur.fetchone()
        
        # Get entity type distribution
        cur.execute(
            """
            SELECT 
                entity->>'label' as entity_type,
                COUNT(*) as count
            FROM structured_documents,
                 jsonb_array_elements(entities) as entity
            GROUP BY entity->>'label'
            ORDER BY count DESC
            """
        )
        entity_types = cur.fetchall()
        
        return {
            'total_documents': basic_stats[0] if basic_stats else 0,
            'avg_entities_per_document': float(basic_stats[1]) if basic_stats and basic_stats[1] else 0,
            'unique_files': basic_stats[2] if basic_stats else 0,
            'entity_types': [{'type': row[0], 'count': row[1]} for row in entity_types]
        }
