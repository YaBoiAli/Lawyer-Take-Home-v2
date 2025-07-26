# --- Structured Data Processor Module ---
# This module converts raw OCR text into structured formats with markdown support.
# It handles entity extraction, table detection, signature identification, and metadata extraction.
#
# CHEAT SHEET:
# - StructuredDataProcessor: Main class for text structuring
#   - __init__: Sets up NLP models and LLM client
#   - extract_entities: Extracts people, organizations, dates, etc.
#   - detect_and_format_tables: Converts table-like text to markdown
#   - identify_signatures: Finds and formats signature blocks
#   - structure_document: Main method that processes raw text
#   - enhance_searchability: Adds metadata for better querying
#
# Output Format: Structured dict with markdown content, entities, metadata
#
# ---
import re
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import spacy
from groq import Groq
from dataclasses import dataclass

# Set up logging for markdown processing
logging.basicConfig(level=logging.INFO)
md_logger = logging.getLogger('MarkdownProcessor')

@dataclass
class EntityInfo:
    """Structured representation of an extracted entity."""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 0.0

@dataclass
class StructuredDocument:
    """Complete structured representation of a document."""
    original_text: str
    markdown_content: str
    entities: List[EntityInfo]
    tables: List[str]
    signatures: List[str]
    metadata: Dict[str, Any]
    searchable_terms: List[str]

class StructuredDataProcessor:
    """
    Processes raw OCR text into structured formats with enhanced searchability.
    Supports markdown formatting, entity extraction, and metadata generation.
    """
    
    def __init__(self, groq_api_key: str):
        # Initialize LLM client for advanced text processing
        self.groq_client = Groq(api_key=groq_api_key)
        
        # Load spaCy model for NLP tasks (install with: python -m spacy download en_core_web_sm)
        try:
            self.nlp = spacy.load("en_core_web_sm")
            md_logger.info("✅ spaCy model 'en_core_web_sm' loaded successfully")
        except IOError:
            md_logger.warning("⚠️ spaCy model 'en_core_web_sm' not found. Entity extraction will be limited.")
            self.nlp = None
        
        # Common patterns for document structure detection
        self.table_patterns = [
            r'^\s*\|.*\|.*$',  # Pipe-separated tables
            r'^\s*[\w\s]+\s+[\d\.\$]+\s*$',  # Name/value pairs with numbers
            r'^\s*\d+[\.\)]\s+.*$',  # Numbered lists
        ]
        
        self.signature_patterns = [
            r'(?i)signature[s]?\s*[:]\s*(.+)',
            r'(?i)signed\s*[:]\s*(.+)',
            r'(?i)by\s*[:]\s*([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'([A-Z][a-z]+\s+[A-Z]\.?\s+[A-Z][a-z]+)',  # Name patterns like "Miguel A Vasquez"
        ]
        
        md_logger.info("🚀 StructuredDataProcessor initialized with markdown logging enabled")

    def extract_entities(self, text: str) -> List[EntityInfo]:
        """
        Extracts named entities (people, organizations, dates, etc.) from text.
        Uses both spaCy NLP and pattern matching for comprehensive extraction.
        """
        md_logger.info("🔍 Starting entity extraction...")
        entities = []
        
        if self.nlp:
            # Use spaCy for standard entity recognition
            md_logger.debug("Using spaCy for NLP-based entity extraction")
            doc = self.nlp(text)
            spacy_entities = 0
            for ent in doc.ents:
                entities.append(EntityInfo(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.8  # Default confidence for spaCy
                ))
                spacy_entities += 1
            md_logger.info(f"📊 spaCy extracted {spacy_entities} entities")
        
        # Additional pattern-based entity extraction for legal/business documents
        md_logger.debug("Running pattern-based entity extraction")
        pattern_entities = 0
        
        # Names with middle initials (common in legal docs)
        name_pattern = r'\b[A-Z][a-z]+\s+[A-Z]\.?\s+[A-Z][a-z]+\b'
        for match in re.finditer(name_pattern, text):
            entities.append(EntityInfo(
                text=match.group(),
                label="PERSON",
                start=match.start(),
                end=match.end(),
                confidence=0.7
            ))
            pattern_entities += 1
            md_logger.debug(f"Found person name: {match.group()}")
        
        # Phone numbers
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        phone_count = 0
        for match in re.finditer(phone_pattern, text):
            entities.append(EntityInfo(
                text=match.group(),
                label="PHONE",
                start=match.start(),
                end=match.end(),
                confidence=0.9
            ))
            phone_count += 1
            md_logger.debug(f"Found phone number: {match.group()}")
        
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_count = 0
        for match in re.finditer(email_pattern, text):
            entities.append(EntityInfo(
                text=match.group(),
                label="EMAIL",
                start=match.start(),
                end=match.end(),
                confidence=0.95
            ))
            email_count += 1
            md_logger.debug(f"Found email: {match.group()}")
        
        # Remove duplicates and sort by position
        unique_entities = []
        seen = set()
        for entity in sorted(entities, key=lambda x: x.start):
            key = (entity.text.lower(), entity.label)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        md_logger.info(f"✅ Entity extraction complete: {len(unique_entities)} unique entities found")
        md_logger.info(f"   📞 Phones: {phone_count}, 📧 Emails: {email_count}, 👤 Pattern entities: {pattern_entities}")
        
        return unique_entities

    def detect_and_format_tables(self, text: str) -> List[str]:
        """
        Detects table-like structures in text and converts them to markdown format.
        Handles various table formats including aligned columns and key-value pairs.
        """
        md_logger.info("📋 Starting table detection and markdown formatting...")
        tables = []
        lines = text.split('\n')
        current_table = []
        in_table = False
        table_count = 0
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                if in_table and current_table:
                    # End of table, process it
                    table_md = self._format_table_as_markdown(current_table)
                    if table_md:
                        table_count += 1
                        tables.append(table_md)
                        md_logger.debug(f"✅ Table {table_count} converted to markdown ({len(current_table)} rows)")
                    current_table = []
                    in_table = False
                continue
            
            # Check if line looks like a table row
            is_table_row = any(re.match(pattern, line) for pattern in self.table_patterns)
            
            # Also check for aligned columns (multiple spaces between words)
            if not is_table_row and len(re.split(r'\s{2,}', line)) > 2:
                is_table_row = True
                md_logger.debug(f"Line {line_num} detected as table row (aligned columns): {line[:50]}...")
            
            if is_table_row:
                if not in_table:
                    md_logger.debug(f"Starting new table at line {line_num}")
                in_table = True
                current_table.append(line)
            elif in_table:
                # End of table
                table_md = self._format_table_as_markdown(current_table)
                if table_md:
                    table_count += 1
                    tables.append(table_md)
                    md_logger.debug(f"✅ Table {table_count} converted to markdown ({len(current_table)} rows)")
                current_table = []
                in_table = False
        
        # Handle table at end of text
        if current_table:
            table_md = self._format_table_as_markdown(current_table)
            if table_md:
                table_count += 1
                tables.append(table_md)
                md_logger.debug(f"✅ Final table {table_count} converted to markdown ({len(current_table)} rows)")
        
        md_logger.info(f"📊 Table detection complete: {len(tables)} tables converted to markdown")
        
        return tables

    def _format_table_as_markdown(self, table_lines: List[str]) -> str:
        """
        Converts a list of table lines into markdown table format.
        Handles various input formats and normalizes them.
        """
        if len(table_lines) < 2:
            return ""
        
        md_logger.debug(f"Converting {len(table_lines)} lines to markdown table")
        
        # Try to detect column structure
        processed_rows = []
        
        for line in table_lines:
            # Split on multiple spaces or pipes
            if '|' in line:
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                md_logger.debug(f"Pipe-separated row: {len(cells)} columns")
            else:
                cells = re.split(r'\s{2,}', line.strip())
                md_logger.debug(f"Space-separated row: {len(cells)} columns")
            
            if cells:
                processed_rows.append(cells)
        
        if not processed_rows:
            return ""
        
        # Ensure all rows have the same number of columns
        max_cols = max(len(row) for row in processed_rows)
        for row in processed_rows:
            while len(row) < max_cols:
                row.append("")
        
        # Build markdown table
        markdown_lines = []
        
        # Header row
        header = "| " + " | ".join(processed_rows[0]) + " |"
        markdown_lines.append(header)
        
        # Separator row
        separator = "| " + " | ".join(["---"] * max_cols) + " |"
        markdown_lines.append(separator)
        
        # Data rows
        for row in processed_rows[1:]:
            row_md = "| " + " | ".join(row) + " |"
            markdown_lines.append(row_md)
        
        result = "\n".join(markdown_lines)
        md_logger.debug(f"✅ Markdown table created: {max_cols} columns, {len(processed_rows)} rows")
        
        return result

    def identify_signatures(self, text: str) -> List[str]:
        """
        Identifies and extracts signature blocks and signatory information.
        Returns formatted signature information.
        """
        md_logger.info("✍️ Starting signature identification...")
        signatures = []
        signature_count = 0
        
        for pattern in self.signature_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                signature_text = match.group().strip()
                if len(signature_text) > 3:  # Filter out very short matches
                    signatures.append(f"**Signature:** {signature_text}")
                    signature_count += 1
                    md_logger.debug(f"Found signature pattern: {signature_text}")
        
        # Look for signature blocks (lines with "Date:" followed by names)
        date_sig_pattern = r'Date\s*[:\-]\s*([^\n]+)\s*\n\s*([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)'
        date_signatures = 0
        for match in re.finditer(date_sig_pattern, text, re.MULTILINE | re.IGNORECASE):
            date_part = match.group(1).strip()
            name_part = match.group(2).strip()
            signatures.append(f"**Signed:** {name_part} **Date:** {date_part}")
            date_signatures += 1
            md_logger.debug(f"Found date signature: {name_part} on {date_part}")
        
        unique_signatures = list(set(signatures))  # Remove duplicates
        md_logger.info(f"✅ Signature identification complete: {len(unique_signatures)} unique signatures found")
        md_logger.info(f"   📝 Pattern signatures: {signature_count}, 📅 Date signatures: {date_signatures}")
        
        return unique_signatures

    def enhance_searchability(self, text: str, entities: List[EntityInfo]) -> List[str]:
        """
        Creates searchable terms and phrases for better query matching.
        Includes entity variations, common misspellings, and related terms.
        """
        md_logger.debug("🔍 Enhancing searchability with term variations...")
        searchable_terms = []
        
        # Add all entity texts
        for entity in entities:
            searchable_terms.append(entity.text.lower())
            
            # Add variations for person names
            if entity.label == "PERSON":
                name_parts = entity.text.split()
                if len(name_parts) >= 2:
                    # Add first and last name combinations
                    searchable_terms.append(f"{name_parts[0]} {name_parts[-1]}".lower())
                    # Add just first name and just last name
                    searchable_terms.extend([part.lower() for part in name_parts])
        
        # Add important keywords from text
        important_words = re.findall(r'\b[A-Z][a-z]{3,}\b', text)  # Capitalized words
        searchable_terms.extend([word.lower() for word in important_words])
        
        # Remove duplicates and short terms
        searchable_terms = list(set(term for term in searchable_terms if len(term) > 2))
        
        md_logger.debug(f"✅ Generated {len(searchable_terms)} searchable terms")
        
        return searchable_terms

    def structure_document(self, ocr_data: Dict[str, Any]) -> StructuredDocument:
        """
        Main method that processes raw OCR data into a structured document.
        Combines all processing steps to create a comprehensive structured output.
        """
        # Extract basic information
        original_text = ocr_data.get('full_text', '')
        file_path = ocr_data.get('file_path', '')
        page_number = ocr_data.get('page_number', 1)
        
        md_logger.info(f"🚀 Starting structured processing for: {file_path} (Page {page_number})")
        md_logger.info(f"📄 Document length: {len(original_text)} characters")
        
        if not original_text.strip():
            # Return empty structure for blank documents
            md_logger.warning("⚠️ No text content detected in document")
            return StructuredDocument(
                original_text=original_text,
                markdown_content="*[No text content detected]*",
                entities=[],
                tables=[],
                signatures=[],
                metadata={
                    'file_name': file_path,
                    'page_number': page_number,
                    'processed_at': datetime.now().isoformat(),
                    'has_content': False
                },
                searchable_terms=[]
            )
        
        # Extract entities
        entities = self.extract_entities(original_text)
        
        # Detect tables
        tables = self.detect_and_format_tables(original_text)
        
        # Identify signatures
        signatures = self.identify_signatures(original_text)
        
        # Create markdown content
        md_logger.info("📝 Creating structured markdown content...")
        markdown_content = self._create_markdown_content(original_text, tables, signatures, entities)
        
        # Generate searchable terms
        searchable_terms = self.enhance_searchability(original_text, entities)
        
        # Create metadata
        metadata = {
            'file_name': file_path,
            'page_number': page_number,
            'processed_at': datetime.now().isoformat(),
            'has_content': True,
            'entity_count': len(entities),
            'table_count': len(tables),
            'signature_count': len(signatures),
            'word_count': len(original_text.split()),
            'entities_by_type': self._group_entities_by_type(entities)
        }
        
        md_logger.info("✅ Structured document processing complete!")
        md_logger.info(f"📊 Final stats: {len(entities)} entities, {len(tables)} tables, {len(signatures)} signatures")
        md_logger.info(f"📝 Markdown content length: {len(markdown_content)} characters")
        
        return StructuredDocument(
            original_text=original_text,
            markdown_content=markdown_content,
            entities=entities,
            tables=tables,
            signatures=signatures,
            metadata=metadata,
            searchable_terms=searchable_terms
        )

    def _create_markdown_content(self, text: str, tables: List[str], signatures: List[str], entities: List[EntityInfo]) -> str:
        """
        Creates formatted markdown content from the original text and extracted elements.
        Enhances readability and structure.
        """
        md_logger.debug("🏗️ Building markdown structure...")
        markdown_parts = []
        
        # Add entity summary if there are many entities
        if len(entities) > 5:
            md_logger.debug("Adding entity summary section")
            markdown_parts.append("## Key Information")
            entity_summary = []
            for entity in entities[:10]:  # Show top 10 entities
                entity_summary.append(f"- **{entity.label}**: {entity.text}")
            markdown_parts.append("\n".join(entity_summary))
            markdown_parts.append("")
        
        # Add tables if found
        if tables:
            md_logger.debug(f"Adding {len(tables)} tables to markdown")
            markdown_parts.append("## Tables")
            for i, table in enumerate(tables, 1):
                markdown_parts.append(f"### Table {i}")
                markdown_parts.append(table)
                markdown_parts.append("")
        
        # Add signatures if found
        if signatures:
            md_logger.debug(f"Adding {len(signatures)} signatures to markdown")
            markdown_parts.append("## Signatures")
            markdown_parts.extend(signatures)
            markdown_parts.append("")
        
        # Add main content
        markdown_parts.append("## Document Content")
        
        # Format the main text with better paragraph breaks
        paragraphs = re.split(r'\n\s*\n', text.strip())
        formatted_paragraphs = []
        header_count = 0
        
        for para in paragraphs:
            para = para.strip()
            if para:
                # Check if paragraph looks like a header (short, capitalized)
                if len(para) < 100 and para.isupper():
                    formatted_paragraphs.append(f"### {para.title()}")
                    header_count += 1
                    md_logger.debug(f"Converted to header: {para.title()}")
                else:
                    formatted_paragraphs.append(para)
        
        markdown_parts.extend(formatted_paragraphs)
        
        final_markdown = "\n\n".join(markdown_parts)
        md_logger.debug(f"✅ Markdown structure complete: {header_count} headers, {len(formatted_paragraphs)} paragraphs")
        
        return final_markdown

    def _group_entities_by_type(self, entities: List[EntityInfo]) -> Dict[str, List[str]]:
        """Groups entities by their type for metadata."""
        grouped = {}
        for entity in entities:
            if entity.label not in grouped:
                grouped[entity.label] = []
            if entity.text not in grouped[entity.label]:
                grouped[entity.label].append(entity.text)
        return grouped

    def process_batch(self, ocr_output: List[Dict[str, Any]]) -> List[StructuredDocument]:
        """
        Processes a batch of OCR output into structured documents.
        Useful for processing multiple pages or documents at once.
        """
        md_logger.info(f"🔄 Starting batch processing of {len(ocr_output)} documents...")
        structured_docs = []
        
        for i, ocr_data in enumerate(ocr_output, 1):
            try:
                md_logger.info(f"📄 Processing document {i}/{len(ocr_output)}: {ocr_data.get('file_path', 'unknown')}")
                structured_doc = self.structure_document(ocr_data)
                structured_docs.append(structured_doc)
                md_logger.info(f"✅ Document {i} processed successfully")
            except Exception as e:
                md_logger.error(f"❌ Error processing document {i} ({ocr_data.get('file_path', 'unknown')}): {e}")
                # Create minimal structure for failed processing
                structured_docs.append(StructuredDocument(
                    original_text=ocr_data.get('full_text', ''),
                    markdown_content=f"*[Processing failed: {str(e)}]*",
                    entities=[],
                    tables=[],
                    signatures=[],
                    metadata={
                        'file_name': ocr_data.get('file_path', ''),
                        'page_number': ocr_data.get('page_number', 1),
                        'processed_at': datetime.now().isoformat(),
                        'has_content': False,
                        'processing_error': str(e)
                    },
                    searchable_terms=[]
                ))
        
        md_logger.info(f"🎉 Batch processing complete: {len(structured_docs)} documents processed")
        return structured_docs 