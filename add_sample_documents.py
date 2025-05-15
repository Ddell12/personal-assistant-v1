#!/usr/bin/env python
"""
Script to add sample documents with embeddings to the Supabase database.
"""
import os
import json
import time
import sys
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not found in environment variables")
    sys.exit(1)
    
openai_client = OpenAI(api_key=api_key)

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
if not supabase_url or not supabase_key:
    print("ERROR: SUPABASE_URL or SUPABASE_SERVICE_KEY not found in environment variables")
    sys.exit(1)
    
supabase = create_client(supabase_url, supabase_key)

# Sample documents with varied content
SAMPLE_DOCUMENTS = [
    {
        "doc_id": "company_mission",
        "content": "Our mission is to create innovative technology solutions that improve people's lives while maintaining the highest standards of ethical conduct and environmental responsibility.",
        "metadata": {"category": "corporate", "importance": "high"}
    },
    {
        "doc_id": "product_roadmap",
        "content": "The Q3 product roadmap includes the launch of our mobile app redesign, integration with third-party payment providers, and the beta release of our AI-powered recommendation engine.",
        "metadata": {"category": "product", "quarter": "Q3", "year": 2025}
    },
    {
        "doc_id": "customer_feedback",
        "content": "Users have reported that the new dashboard interface is intuitive and helps them find information more quickly. However, some have requested more customization options and better mobile responsiveness.",
        "metadata": {"category": "feedback", "source": "user survey"}
    },
    {
        "doc_id": "technical_spec",
        "content": "The authentication service will use OAuth 2.0 with JWT tokens for stateless authentication. Token expiration is set to 1 hour, with refresh tokens valid for 30 days. All sensitive data must be encrypted at rest using AES-256.",
        "metadata": {"category": "technical", "team": "security"}
    },
    {
        "doc_id": "marketing_campaign",
        "content": "The summer marketing campaign will focus on outdoor activities and how our products enhance the experience. Key messaging should emphasize durability, weather resistance, and long battery life.",
        "metadata": {"category": "marketing", "season": "summer"}
    },
    {
        "doc_id": "research_findings",
        "content": "Our research indicates that 78% of users prefer voice commands for hands-free operation, especially while driving or cooking. Accuracy and natural language understanding were cited as the most important factors.",
        "metadata": {"category": "research", "method": "user interviews"}
    },
    {
        "doc_id": "employee_handbook",
        "content": "Remote work policy: Employees may work remotely up to 3 days per week with manager approval. Core hours are 10am-3pm in the employee's local time zone. All team meetings should accommodate various time zones when possible.",
        "metadata": {"category": "HR", "last_updated": "2025-03-15"}
    },
    {
        "doc_id": "financial_report",
        "content": "Q1 financial highlights: Revenue increased 15% year-over-year, with subscription services growing 22%. Operating expenses were reduced by 5% due to improved cloud infrastructure efficiency. Cash reserves remain strong at $24.5M.",
        "metadata": {"category": "finance", "quarter": "Q1", "year": 2025}
    },
    {
        "doc_id": "competitive_analysis",
        "content": "Competitor X recently launched a similar feature but lacks the machine learning capabilities that differentiate our product. Their pricing is 15% lower, but our customer retention rate is 23% higher according to industry reports.",
        "metadata": {"category": "strategy", "confidential": True}
    },
    {
        "doc_id": "sustainability_initiative",
        "content": "Our green office initiative aims to reduce paper consumption by 90% and energy usage by 30% by the end of 2025. All new office equipment must meet Energy Star certification, and we're transitioning to 100% renewable energy sources.",
        "metadata": {"category": "sustainability", "goal": "carbon neutral"}
    }
]

def generate_embedding(text):
    """Generate an embedding for the given text using OpenAI's API."""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return None

def add_document_to_database(doc_id, content, metadata, embedding):
    """Add a document with its embedding to the Supabase database."""
    try:
        # Prepare the document data
        document = {
            "doc_id": doc_id,
            "content": content,
            "metadata": json.dumps(metadata),
            "embedding": embedding
        }
        
        # Insert the document
        response = supabase.from_("documents").upsert(document).execute()
        
        if hasattr(response, 'error') and response.error:
            print(f"Error adding document {doc_id}: {response.error}")
            return False
            
        return True
    except Exception as e:
        print(f"Error adding document {doc_id}: {str(e)}")
        return False

def main():
    """Add sample documents to the database."""
    print(f"Adding {len(SAMPLE_DOCUMENTS)} sample documents to the database...")
    
    success_count = 0
    
    for doc in SAMPLE_DOCUMENTS:
        print(f"Processing document: {doc['doc_id']}")
        
        # Generate embedding for the document
        embedding = generate_embedding(doc['content'])
        if not embedding:
            print(f"Skipping document {doc['doc_id']} due to embedding generation failure")
            continue
            
        # Add the document to the database
        if add_document_to_database(doc['doc_id'], doc['content'], doc['metadata'], embedding):
            success_count += 1
            print(f"Successfully added document: {doc['doc_id']}")
        else:
            print(f"Failed to add document: {doc['doc_id']}")
            
        # Sleep briefly to avoid rate limits
        time.sleep(0.5)
    
    print(f"\nAdded {success_count} out of {len(SAMPLE_DOCUMENTS)} documents to the database")

if __name__ == "__main__":
    main()
