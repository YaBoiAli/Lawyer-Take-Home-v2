
import psycopg2
import uuid
from datetime import datetime

def get_db_connection(neon_connection_string):
    """Establishes a connection to the Neon database."""
    try:
        conn = psycopg2.connect(neon_connection_string)
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to Neon DB: {e}")
        return None

def create_documents_table(conn):
    """Creates the documents table, pg_trgm extension, and a GIN index."""
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


def insert_ocr_data(conn, ocr_output):
    """Inserts OCR data into the documents table, avoiding duplicates."""
    with conn.cursor() as cur:
        cur.execute("SELECT file_name, page_number FROM documents;")
        existing_records = {(row[0], row[1]) for row in cur.fetchall()}

        records_to_insert = []
        for page_data in ocr_output:
            file_name = page_data.get("file_path", "").split("/")[-1]
            page_number = page_data.get("page_number")
            page_text = page_data.get("full_text")

            if (file_name, page_number) not in existing_records:
                records_to_insert.append((str(uuid.uuid4()), file_name, page_number, page_text))

        if records_to_insert:
            cur.executemany(
                "INSERT INTO documents (file_id, file_name, page_number, page_text) VALUES (%s, %s, %s, %s)",
                records_to_insert
            )
            conn.commit()
            print(f"{len(records_to_insert)} new records inserted successfully into Neon DB.")
        else:
            print("No new records to insert into Neon DB. Data may already exist.")

def fuzzy_search_all_documents(conn, search_query, similarity_threshold=0.3):
    """
    Performs a case-insensitive fuzzy search across ALL documents in the database.
    """
    results = []
    with conn.cursor() as cur:
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
