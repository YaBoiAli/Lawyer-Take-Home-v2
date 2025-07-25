
import re
import yake
from neo4j import GraphDatabase
from groq import Groq
from sentence_transformers import SentenceTransformer

class Neo4jRAGSystem:
    """
    A complete RAG system using Neo4j for graph storage, free embeddings,
    and Groq for the LLM.
    """
    def __init__(self, neo4j_uri, neo4j_user, neo4j_pass, groq_api_key):
        """
        Initializes the connection to databases and loads the necessary models.
        """
        try:
            print(f"Attempting to connect to Neo4j with URI: {neo4j_uri}, User: {neo4j_user}")
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
            self.driver.verify_connectivity()
            print("Successfully connected to Neo4j.")
        except Exception as e:
            import traceback
            print(f"Failed to connect to Neo4j: {e}")
            print("Neo4j connection traceback:")
            traceback.print_exc() 
            self.driver = None
            return
        self.groq_client = Groq(api_key=groq_api_key)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=10)
        self._create_graph_schema()

    def _create_graph_schema(self):
        """
        Ensures the necessary constraints and vector index exist in the Neo4j database.
        """
        if not self.driver: return
        print("Setting up Neo4j graph schema (constraints and vector index)...")
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
            session.run("CREATE CONSTRAINT keyword_name_unique IF NOT EXISTS FOR (k:Keyword) REQUIRE k.name IS UNIQUE")
            session.run("""
                CREATE VECTOR INDEX `chunk_embedding_index` IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS { indexConfig: {
                    `vector.dimensions`: 384,
                    `vector.similarity_function`: 'cosine'
                }}
            """)
        print("Neo4j graph schema is ready.")

    def ingest_data(self, ocr_output):
        """
        Processes OCR data, generates embeddings, and stores it in the Neo4j graph.
        """
        if not self.driver:
            print("Cannot ingest data: No database connection.")
            return
        
        with self.driver.session() as session:
            for page_data in ocr_output:
                file_name = page_data.get("file_path", "unknown.png").split("/")[-1]
                page_num = page_data.get("page_number")
                text = page_data.get("full_text", "").replace('\n', ' ')
                
                if not text or not page_num: continue
                chunk_id = f"{file_name}_p{page_num}"
                embedding = self.embedding_model.encode(text).tolist()
                keywords = [kw[0].lower() for kw in self.kw_extractor.extract_keywords(text)]
                query = """
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $text, c.file_name = $file_name, c.page_number = $page_num, c.embedding = $embedding
                WITH c UNWIND $keywords AS keyword_text
                MERGE (k:Keyword {name: keyword_text})
                MERGE (c)-[:MENTIONS]->(k)
                """
                session.run(query, chunk_id=chunk_id, text=text, file_name=file_name, page_num=page_num, embedding=embedding, keywords=keywords)
                print(f"   ingested chunk into Neo4j: {chunk_id}")
        print("--- Neo4j Data Ingestion Complete ---")

    def retrieve(self, query, top_k=3, file_name=None):
        """
        Finds the most relevant text chunks from Neo4j using vector similarity search.
        Optionally filters by a specific file name.
        """
        if not self.driver: return []
        query_embedding = self.embedding_model.encode(query).tolist()
        
        cypher_query = """
            CALL db.index.vector.queryNodes('chunk_embedding_index', $top_k, $query_embedding)
            YIELD node AS chunk, score
        """
        params = {"top_k": top_k, "query_embedding": query_embedding}

        if file_name:
            cypher_query += " WHERE chunk.file_name = $file_name"
            params["file_name"] = file_name
            
        cypher_query += " RETURN chunk.text AS text, chunk.file_name AS file_name, chunk.page_number AS page_number, score"

        with self.driver.session() as session:
            result = session.run(cypher_query, params)
            return [record for record in result]

    def generate_answer(self, query, file_name=None):
        """
        Performs the full RAG process: retrieves context and generates an answer using Groq.
        Optionally filters retrieval by a specific file name.
        """
        retrieved_chunks = self.retrieve(query, file_name=file_name)
        
        if not retrieved_chunks:
            return "I could not find any relevant information in the documents to answer your question."

        MAX_CONTEXT_CHARS = 7000 # Approximately 7000 characters for context, leaving room for prompt and answer
        
        context_parts = []
        current_context_length = 0

        for record in retrieved_chunks:
            source_info = f"Source: {record['file_name']} (Page: {record['page_number']})"
            chunk_text = record['text']
            
            # Estimate length of the chunk with source info
            chunk_with_source = f"{chunk_text}\n[{source_info}]"
            
            # If adding this chunk exceeds the max context length, stop
            # Add a buffer for the separator "\n\n---\n\n"
            if current_context_length + len(chunk_with_source) + len("\n\n---\n\n") > MAX_CONTEXT_CHARS:
                print(f"Stopping context accumulation due to length limit. Current: {current_context_length}, Adding: {len(chunk_with_source)}")
                break
            
            context_parts.append(chunk_with_source)
            current_context_length += len(chunk_with_source) + len("\n\n---\n\n") # Add separator length for next iteration
            
        context_str = "\n\n---\n\n".join(context_parts)
        
        if not context_str:
            return "I could not find enough relevant information within the allowed context size to answer your question."

        prompt = f"""
        You are a helpful assistant who answers questions based ONLY on the context provided.
        Each piece of context is followed by its source in brackets, e.g., [Source: document.pdf (Page: 1)].
        Do not use any outside information. If the answer is not in the context, say so.
        Always cite the source (file name and page number) for each piece of information you use in your answer.
        
        CONTEXT: {context_str}
        
        QUESTION: {query}
        
        ANSWER:
        """
        
        chat_completion = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
        )
        return chat_completion.choices[0].message.content

    def close(self):
        """Closes the database connection."""
        if self.driver is not None:
            self.driver.close()
            print("\nNeo4j connection closed.")
