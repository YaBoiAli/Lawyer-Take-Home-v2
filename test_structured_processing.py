#!/usr/bin/env python3
"""
Test script to demonstrate structured data processing capabilities.
This script shows how OCR text gets converted to structured format with:
- Entity extraction (people, organizations, dates, etc.)
- Table detection and markdown formatting
- Signature identification
- Enhanced searchability

Run this script to see examples of the structured output.
"""

import os
import sys
from modules.structured_data_processor import StructuredDataProcessor

# Sample OCR data that might come from a legal document
SAMPLE_OCR_DATA = [
    {
        "file_path": "contract_sample.pdf",
        "page_number": 1,
        "full_text": """
        EMPLOYMENT AGREEMENT
        
        This Employment Agreement is entered into on March 15, 2024, between
        ACME Corporation, a Delaware corporation ("Company"), and Miguel A Vasquez
        ("Employee").
        
        Employee Information:
        Name:           Miguel A Vasquez
        Address:        123 Main Street, Anytown, CA 90210
        Phone:          555-123-4567
        Email:          miguel.vasquez@email.com
        Position:       Senior Software Engineer
        Start Date:     April 1, 2024
        Salary:         $120,000 annually
        
        Compensation Details:
        Base Salary     $120,000
        Bonus          Up to 20% of base
        Stock Options  5,000 shares
        Benefits       Health, Dental, Vision
        
        SIGNATURES:
        
        Date: March 15, 2024
        Miguel A Vasquez
        Employee
        
        Date: March 15, 2024
        Sarah Johnson
        CEO, ACME Corporation
        """
    },
    {
        "file_path": "invoice_sample.pdf", 
        "page_number": 1,
        "full_text": """
        INVOICE #INV-2024-001
        
        Bill To:
        Miguel A Vasquez
        Vasquez Consulting LLC
        456 Business Ave
        Suite 200
        Business City, NY 10001
        
        Invoice Date: February 28, 2024
        Due Date: March 30, 2024
        
        Services Provided:
        Description                    Hours    Rate      Total
        Software Development           40       $150      $6,000
        System Architecture Review     8        $200      $1,600
        Code Review & Testing          12       $125      $1,500
        
        Subtotal:                                        $9,100
        Tax (8.25%):                                     $750.75
        Total:                                           $9,850.75
        
        Payment Terms: Net 30 days
        Remit to: payments@consulting.com
        """
    }
]

def test_structured_processing():
    """Test the structured data processing with sample data."""
    
    # You'll need to set your GROQ_API_KEY environment variable
    groq_api_key = os.environ.get('GROQ_API_KEY')
    if not groq_api_key:
        print("Error: Please set the GROQ_API_KEY environment variable")
        print("You can get a free API key from: https://console.groq.com/")
        return
    
    print("🚀 Testing Structured Data Processing")
    print("=" * 50)
    
    # Initialize the processor
    processor = StructuredDataProcessor(groq_api_key)
    
    # Process each sample document
    for i, ocr_data in enumerate(SAMPLE_OCR_DATA, 1):
        print(f"\n📄 Processing Document {i}: {ocr_data['file_path']}")
        print("-" * 40)
        
        try:
            # Process the document
            structured_doc = processor.structure_document(ocr_data)
            
            # Display results
            print(f"✅ Processing completed successfully!")
            print(f"📊 Statistics:")
            print(f"   - Entities found: {len(structured_doc.entities)}")
            print(f"   - Tables detected: {len(structured_doc.tables)}")
            print(f"   - Signatures found: {len(structured_doc.signatures)}")
            print(f"   - Searchable terms: {len(structured_doc.searchable_terms)}")
            
            # Show entities by type
            if structured_doc.entities:
                print(f"\n🏷️  Extracted Entities:")
                entity_types = {}
                for entity in structured_doc.entities:
                    if entity.label not in entity_types:
                        entity_types[entity.label] = []
                    entity_types[entity.label].append(entity.text)
                
                for entity_type, entities in entity_types.items():
                    print(f"   {entity_type}: {', '.join(set(entities))}")
            
            # Show tables
            if structured_doc.tables:
                print(f"\n📋 Detected Tables:")
                for j, table in enumerate(structured_doc.tables, 1):
                    print(f"   Table {j}:")
                    print("   " + "\n   ".join(table.split('\n')[:5]))  # Show first 5 lines
                    if len(table.split('\n')) > 5:
                        print("   ...")
            
            # Show signatures
            if structured_doc.signatures:
                print(f"\n✍️  Signatures:")
                for signature in structured_doc.signatures:
                    print(f"   {signature}")
            
            # Show searchable terms (first 10)
            print(f"\n🔍 Searchable Terms (sample):")
            sample_terms = structured_doc.searchable_terms[:10]
            print(f"   {', '.join(sample_terms)}")
            if len(structured_doc.searchable_terms) > 10:
                print(f"   ... and {len(structured_doc.searchable_terms) - 10} more")
            
            # Show markdown preview
            print(f"\n📝 Markdown Content Preview:")
            markdown_lines = structured_doc.markdown_content.split('\n')
            preview_lines = markdown_lines[:15]  # Show first 15 lines
            for line in preview_lines:
                print(f"   {line}")
            if len(markdown_lines) > 15:
                print("   ...")
            
        except Exception as e:
            print(f"❌ Error processing document: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎯 Example Entity Queries You Can Now Make:")
    print("   - 'Find all info on Miguel A Vasquez'")
    print("   - 'Show me documents mentioning ACME Corporation'")
    print("   - 'What contracts were signed in March 2024?'")
    print("   - 'Find all invoices with payment terms'")
    print("   - 'Show salary information for software engineers'")
    
    print(f"\n✨ Structured Processing Complete!")
    print("The processed documents now support:")
    print("   ✓ Entity-based search (people, companies, dates)")
    print("   ✓ Markdown-formatted tables")
    print("   ✓ Signature block identification")
    print("   ✓ Enhanced searchability with variations")
    print("   ✓ Metadata-rich storage for advanced queries")

def demonstrate_entity_search():
    """Demonstrate how entity-based searching would work."""
    print(f"\n🔍 Entity Search Examples:")
    print("=" * 30)
    
    # This would be the type of queries you can now make
    example_queries = [
        {
            "query": "Miguel A Vasquez",
            "description": "Find all documents mentioning this person",
            "expected": "Employment contract, Invoice - comprehensive profile"
        },
        {
            "query": "ACME Corporation", 
            "description": "Find all documents related to this company",
            "expected": "Employment agreement as employer"
        },
        {
            "query": "March 2024",
            "description": "Find documents from this time period", 
            "expected": "Employment agreement signed in March"
        },
        {
            "query": "Software Engineer",
            "description": "Find job-related documents",
            "expected": "Employment contract with position details"
        }
    ]
    
    for i, example in enumerate(example_queries, 1):
        print(f"{i}. Query: '{example['query']}'")
        print(f"   Purpose: {example['description']}")
        print(f"   Expected: {example['expected']}")
        print()

if __name__ == "__main__":
    test_structured_processing()
    demonstrate_entity_search() 