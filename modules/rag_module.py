
# --- Neo4j RAG System Module ---
# This module implements a Retrieval-Augmented Generation (RAG) system using Neo4j for graph storage and vector search.
# It manages ingestion of OCR data, vector embedding, keyword extraction, and retrieval for LLM-based Q&A.
#
# CHEAT SHEET:
# - Neo4jRAGSystem: Main RAG class
#   - __init__: Connects to Neo4j, sets up models
#   - _create_graph_schema: Ensures constraints and vector index
#   - ingest_data: Stores OCR data as nodes with embeddings/keywords
#   - ingest_structured_data: Stores structured documents with entities
#   - retrieve: Vector similarity search for relevant chunks
#   - retrieve_by_entity: Entity-aware retrieval for specific people/orgs
#   - generate_answer: Full RAG pipeline (retrieve + LLM answer)
#   - generate_structured_answer: Enhanced answer with markdown formatting
#   - close: Closes Neo4j connection
#
# Node: Chunk(id, text, file_name, page_number, embedding)
# Node: StructuredChunk(id, original_text, markdown_content, entities, metadata, embedding)
# Node: Entity(name, type)  
# Node: Keyword(name)
# Relationship: (Chunk)-[:MENTIONS]->(Keyword)
# Relationship: (StructuredChunk)-[:CONTAINS]->(Entity)
#
# ---
import re
import yake
from neo4j import GraphDatabase
from groq import Groq
from sentence_transformers import SentenceTransformer
import json
from typing import List, Dict, Any, Optional

class Neo4jRAGSystem:
    """
    A complete RAG system using Neo4j for graph storage, free embeddings,
    and Groq for the LLM.
    """
    def __init__(self, neo4j_uri, neo4j_user, neo4j_pass, groq_api_key):
        # Connects to Neo4j, initializes embedding model, keyword extractor, and Groq LLM client.
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
        # Ensures unique constraints and vector index exist in Neo4j for efficient storage and retrieval.
        if not self.driver: return
        print("Setting up Neo4j graph schema (constraints and vector index)...")
        with self.driver.session() as session:
            # Unique constraint for Chunk nodes (id) and Keyword nodes (name)
            session.run("CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
            session.run("CREATE CONSTRAINT keyword_name_unique IF NOT EXISTS FOR (k:Keyword) REQUIRE k.name IS UNIQUE")
            
            # Constraints for structured data
            session.run("CREATE CONSTRAINT structured_chunk_id_unique IF NOT EXISTS FOR (sc:StructuredChunk) REQUIRE sc.id IS UNIQUE")
            session.run("CREATE CONSTRAINT entity_unique IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE")
            
            # Vector index for fast similarity search on embeddings
            session.run("""
                CREATE VECTOR INDEX `chunk_embedding_index` IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS { indexConfig: {
                    `vector.dimensions`: 384,
                    `vector.similarity_function`: 'cosine'
                }}
            """)
            
            # Vector index for structured chunks
            session.run("""
                CREATE VECTOR INDEX `structured_chunk_embedding_index` IF NOT EXISTS
                FOR (sc:StructuredChunk) ON (sc.embedding)
                OPTIONS { indexConfig: {
                    `vector.dimensions`: 384,
                    `vector.similarity_function`: 'cosine'
                }}
            """)
        print("Neo4j graph schema is ready.")

    def ingest_data(self, ocr_output):
        # Ingests OCR output: creates Chunk nodes with embeddings and links to Keyword nodes.
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
                # Generate vector embedding for the text
                embedding = self.embedding_model.encode(text).tolist()
                # Extract keywords using YAKE
                keywords = [kw[0].lower() for kw in self.kw_extractor.extract_keywords(text)]
                # Cypher query: MERGE chunk, set properties, link to keywords
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

    def ingest_structured_data(self, structured_docs):
        # Ingests structured documents: creates StructuredChunk nodes with entities and enhanced metadata.
        if not self.driver:
            print("Cannot ingest structured data: No database connection.")
            return
        
        with self.driver.session() as session:
            for doc in structured_docs:
                file_name = doc.metadata.get('file_name', '').split('/')[-1]
                page_num = doc.metadata.get('page_number', 1)
                
                if not doc.original_text.strip():
                    continue
                    
                chunk_id = f"structured_{file_name}_p{page_num}"
                
                # Generate embedding for the markdown content (better for retrieval)
                embedding = self.embedding_model.encode(doc.markdown_content).tolist()
                
                # Create StructuredChunk node
                chunk_query = """
                MERGE (sc:StructuredChunk {id: $chunk_id})
                SET sc.original_text = $original_text,
                    sc.markdown_content = $markdown_content,
                    sc.file_name = $file_name,
                    sc.page_number = $page_num,
                    sc.embedding = $embedding,
                    sc.metadata = $metadata,
                    sc.searchable_terms = $searchable_terms
                """
                
                session.run(chunk_query, 
                           chunk_id=chunk_id,
                           original_text=doc.original_text,
                           markdown_content=doc.markdown_content,
                           file_name=file_name,
                           page_num=page_num,
                           embedding=embedding,
                           metadata=json.dumps(doc.metadata),
                           searchable_terms=doc.searchable_terms)
                
                # Create Entity nodes and relationships
                for entity in doc.entities:
                    entity_query = """
                    MATCH (sc:StructuredChunk {id: $chunk_id})
                    MERGE (e:Entity {name: $entity_name, type: $entity_type})
                    SET e.confidence = $confidence
                    MERGE (sc)-[:CONTAINS {start: $start, end: $end, confidence: $confidence}]->(e)
                    """
                    
                    session.run(entity_query,
                               chunk_id=chunk_id,
                               entity_name=entity.text,
                               entity_type=entity.label,
                               confidence=entity.confidence,
                               start=entity.start,
                               end=entity.end)
                
                print(f"   ingested structured chunk into Neo4j: {chunk_id} with {len(doc.entities)} entities")
        print("--- Neo4j Structured Data Ingestion Complete ---")

    def retrieve(self, query, top_k=3, file_name=None):
        # Retrieves the most relevant text chunks using vector similarity search.
        # Optionally filters by file_name.
        if not self.driver: return []
        # Generate embedding for the query
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Cypher query for vector search
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

    def retrieve_structured(self, query, top_k=5, file_name=None):
        # Retrieves structured chunks with markdown content and entity information.
        if not self.driver: return []
        
        query_embedding = self.embedding_model.encode(query).tolist()
        
        cypher_query = """
            CALL db.index.vector.queryNodes('structured_chunk_embedding_index', $top_k, $query_embedding)
            YIELD node AS chunk, score
        """
        params = {"top_k": top_k, "query_embedding": query_embedding}

        if file_name:
            cypher_query += " WHERE chunk.file_name = $file_name"
            params["file_name"] = file_name
            
        cypher_query += """
            RETURN chunk.markdown_content AS markdown_content, 
                   chunk.original_text AS original_text,
                   chunk.file_name AS file_name, 
                   chunk.page_number AS page_number, 
                   chunk.metadata AS metadata,
                   score
        """

        with self.driver.session() as session:
            result = session.run(cypher_query, params)
            return [record for record in result]

    def retrieve_by_entity(self, entity_name: str, entity_type: Optional[str] = None, top_k=5):
        # Retrieves documents that contain specific entities (e.g., "Miguel A Vasquez").
        if not self.driver: return []
        
        if entity_type:
            cypher_query = """
                MATCH (sc:StructuredChunk)-[:CONTAINS]->(e:Entity {name: $entity_name, type: $entity_type})
                RETURN sc.markdown_content AS markdown_content,
                       sc.original_text AS original_text,
                       sc.file_name AS file_name,
                       sc.page_number AS page_number,
                       sc.metadata AS metadata,
                       e.name AS entity_name,
                       e.type AS entity_type,
                       1.0 AS score
                ORDER BY sc.file_name, sc.page_number
                LIMIT $top_k
            """
            params = {"entity_name": entity_name, "entity_type": entity_type, "top_k": top_k}
        else:
            cypher_query = """
                MATCH (sc:StructuredChunk)-[:CONTAINS]->(e:Entity)
                WHERE e.name CONTAINS $entity_name
                RETURN sc.markdown_content AS markdown_content,
                       sc.original_text AS original_text,
                       sc.file_name AS file_name,
                       sc.page_number AS page_number,
                       sc.metadata AS metadata,
                       e.name AS entity_name,
                       e.type AS entity_type,
                       1.0 AS score
                ORDER BY sc.file_name, sc.page_number
                LIMIT $top_k
            """
            params = {"entity_name": entity_name, "top_k": top_k}
        
        with self.driver.session() as session:
            result = session.run(cypher_query, params)
            return [record for record in result]

    def generate_answer(self, query, file_name=None):
        # Full RAG pipeline: retrieves context from Neo4j and generates an answer using Groq LLM.
        # Always cites sources and only uses retrieved context.
        retrieved_chunks = self.retrieve(query, file_name=file_name)
        
        if not retrieved_chunks:
            return "I could not find any relevant information in the documents to answer your question."

        MAX_CONTEXT_CHARS = 7000 # Limit context size for LLM prompt
        
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

        # Prompt for the LLM: instructs to use only provided context and always cite sources
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

    def generate_structured_answer(self, query, file_name=None, entity_filter=None):
        # Enhanced RAG pipeline using structured data with markdown formatting and entity awareness.
        # Combines vector search with entity-based retrieval for comprehensive answers.
        
        # Try entity-based retrieval first if entity_filter is provided
        entity_chunks = []
        if entity_filter:
            entity_chunks = self.retrieve_by_entity(entity_filter, top_k=3)
        
        # Get structured chunks via vector search
        vector_chunks = self.retrieve_structured(query, top_k=5, file_name=file_name)
        
        # Combine and deduplicate results
        all_chunks = entity_chunks + vector_chunks
        seen_chunks = set()
        unique_chunks = []
        
        for chunk in all_chunks:
            chunk_key = (chunk['file_name'], chunk['page_number'])
            if chunk_key not in seen_chunks:
                seen_chunks.add(chunk_key)
                unique_chunks.append(chunk)
        
        if not unique_chunks:
            return "I could not find any relevant information in the documents to answer your question."

        MAX_CONTEXT_CHARS = 8000  # Slightly larger for structured content
        
        context_parts = []
        current_context_length = 0

        for record in unique_chunks[:6]:  # Limit to top 6 chunks
            source_info = f"Source: {record['file_name']} (Page: {record['page_number']})"
            
            # Use markdown content if available, otherwise original text
            chunk_text = record.get('markdown_content', record.get('original_text', ''))
            
            chunk_with_source = f"{chunk_text}\n\n[{source_info}]"
            
            if current_context_length + len(chunk_with_source) + len("\n\n---\n\n") > MAX_CONTEXT_CHARS:
                print(f"Stopping context accumulation due to length limit. Current: {current_context_length}, Adding: {len(chunk_with_source)}")
                break
            
            context_parts.append(chunk_with_source)
            current_context_length += len(chunk_with_source) + len("\n\n---\n\n")
            
        context_str = "\n\n---\n\n".join(context_parts)
        
        if not context_str:
            return "I could not find enough relevant information within the allowed context size to answer your question."

        # Enhanced prompt for structured data
        prompt = f"""
        You are a helpful assistant who answers questions based ONLY on the context provided.
        The context includes structured documents with markdown formatting, tables, and entity information.
        Each piece of context is followed by its source in brackets, e.g., [Source: document.pdf (Page: 1)].
        
        Instructions:
        - Use ONLY the provided context - no outside information
        - Always cite sources (file name and page number) for each piece of information
        - Preserve markdown formatting in your response when relevant (tables, headers, etc.)
        - If the context contains tables, include them in your answer
        - If asked about specific people or entities, provide comprehensive information from all relevant documents
        - If the answer is not in the context, say so clearly
        
        CONTEXT: {context_str}
        
        QUESTION: {query}
        
        ANSWER:
        """
        
        chat_completion = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
        )
        return chat_completion.choices[0].message.content

    def get_entity_summary(self, entity_name: str):
        # Gets a comprehensive summary of all information about a specific entity across all documents.
        if not self.driver: return None
        
        with self.driver.session() as session:
            query = """
            MATCH (sc:StructuredChunk)-[r:CONTAINS]->(e:Entity)
            WHERE e.name CONTAINS $entity_name
            RETURN sc.markdown_content AS content,
                   sc.file_name AS file_name,
                   sc.page_number AS page_number,
                   e.name AS entity_name,
                   e.type AS entity_type,
                   r.confidence AS confidence
            ORDER BY sc.file_name, sc.page_number
            """
            
            results = session.run(query, entity_name=entity_name)
            entity_info = []
            
            for record in results:
                entity_info.append({
                    'content': record['content'],
                    'file_name': record['file_name'],
                    'page_number': record['page_number'],
                    'entity_name': record['entity_name'],
                    'entity_type': record['entity_type'],
                    'confidence': record['confidence']
                })
            
            return entity_info

    def close(self):
        # Closes the Neo4j database connection.
        if self.driver is not None:
            self.driver.close()
            print("\nNeo4j connection closed.")
