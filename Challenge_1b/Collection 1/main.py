# -*- coding: utf-8 -*-
"""
Document Intelligence System - PyCharm Version
A system that extracts and prioritizes document sections based on persona and job-to-be-done
"""

import json
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple
import numpy as np
from collections import defaultdict

# Import required libraries
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch


def setup_nltk_data():
    """Download required NLTK data"""
    print(" Downloading required NLTK data...")

    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('omw-1.4', quiet=True)

        print("All NLTK resources downloaded successfully!")

        # Quick test to verify everything works
        test_text = "This is a test sentence. It should work now!"
        sentences = sent_tokenize(test_text)
        words = word_tokenize(test_text)

        print(f" Sentence tokenization works: {len(sentences)} sentences")
        print(f" Word tokenization works: {len(words)} words")
        print(" NLTK is ready to use!")

    except Exception as e:
        print(f" Error downloading NLTK data: {e}")
        print("Please ensure you have internet connection for first-time setup.")


def check_pdf_files(pdf_directory):
    """Check if PDF files exist in the directory"""
    print("Checking if files are uploaded correctly...")
    print()

    if os.path.exists(pdf_directory):
        print(" PDFs folder found!")

        # List all files in PDFs folder
        pdf_files = os.listdir(pdf_directory)
        print(f" Found {len(pdf_files)} files in PDFs folder:")

        for file in pdf_files:
            print(f"    {file}")

        # Check if we have all expected files
        expected_files = [
            "South of France - Cities.pdf",
            "South of France - Cuisine.pdf",
            "South of France - History.pdf",
            "South of France - Restaurants and Hotels.pdf",
            "South of France - Things to Do.pdf",
            "South of France - Tips and Tricks.pdf",
            "South of France - Traditions and Culture.pdf"
        ]

        missing_files = []
        for expected in expected_files:
            if expected not in pdf_files:
                missing_files.append(expected)

        if not missing_files:
            print("\n Perfect! All 7 PDF files are uploaded correctly!")
            print("You're ready to run the document intelligence system!")
            return True
        else:
            print(f"\n Missing files: {missing_files}")
            print("Please upload the missing files to the PDFs folder.")
            return False
    else:
        print(" PDFs folder not found!")
        print(f"Please create a folder named '{pdf_directory}' and upload your PDF files there.")
        return False


def create_input_config(output_path):
    """Create the input configuration file"""
    input_config = {
        "challenge_info": {
            "challenge_id": "round_1b_002",
            "test_case_name": "travel_planner",
            "description": "France Travel"
        },
        "documents": [
            {"filename": "South of France - Cities.pdf", "title": "South of France - Cities"},
            {"filename": "South of France - Cuisine.pdf", "title": "South of France - Cuisine"},
            {"filename": "South of France - History.pdf", "title": "South of France - History"},
            {"filename": "South of France - Restaurants and Hotels.pdf",
             "title": "South of France - Restaurants and Hotels"},
            {"filename": "South of France - Things to Do.pdf", "title": "South of France - Things to Do"},
            {"filename": "South of France - Tips and Tricks.pdf", "title": "South of France - Tips and Tricks"},
            {"filename": "South of France - Traditions and Culture.pdf",
             "title": "South of France - Traditions and Culture"}
        ],
        "persona": {
            "role": "Travel Planner"
        },
        "job_to_be_done": {
            "task": "Plan a trip of 4 days for a group of 10 college friends."
        }
    }

    # Save the configuration
    with open(output_path, 'w') as f:
        json.dump(input_config, f, indent=4)

    print("Input configuration created!")
    return input_config


class DocumentIntelligenceSystem:
    """
    A system that extracts and prioritizes document sections based on persona and job-to-be-done
    """

    def __init__(self):
        """Initialize the system with necessary models and tools"""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Use a lightweight sentence transformer model for semantic similarity
        print("Loading sentence transformer model...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

        # TF-IDF vectorizer for keyword extraction
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )

        print("System initialized successfully!")

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from PDF file page by page
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                document_data = {
                    'filename': os.path.basename(pdf_path),
                    'pages': [],
                    'total_pages': len(pdf_reader.pages)
                }

                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        text = page.extract_text()
                        if text.strip():  # Only add non-empty pages
                            document_data['pages'].append({
                                'page_number': page_num,
                                'text': text.strip(),
                                'sections': self.extract_sections_from_text(text, page_num)
                            })
                    except Exception as e:
                        print(f"Error extracting text from page {page_num}: {str(e)}")
                        continue

                return document_data

        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {str(e)}")
            return None

    def extract_sections_from_text(self, text: str, page_number: int) -> List[Dict]:
        """
        Extract sections from text using various heuristics
        """
        sections = []

        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()

        # Method 1: Look for titles/headers (lines that are short and likely titles)
        lines = text.split('\n')
        current_section = ""
        current_title = ""

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Heuristic for section titles
            is_potential_title = (
                    len(line) < 100 and  # Titles are usually short
                    len(line) > 5 and  # But not too short
                    (line.isupper() or  # All caps
                     line.istitle() or  # Title case
                     re.match(r'^[A-Z][a-zA-Z\s]+$', line) or  # Starts with capital
                     any(keyword in line.lower() for keyword in ['guide', 'tips', 'how to', 'chapter', 'section']))
            )

            if is_potential_title and current_section:
                # Save previous section
                sections.append({
                    'title': current_title or f"Section {len(sections) + 1}",
                    'content': current_section.strip(),
                    'page_number': page_number,
                    'start_position': max(0, i - len(current_section.split('\n')))
                })
                current_section = ""
                current_title = line
            else:
                if is_potential_title and not current_title:
                    current_title = line
                else:
                    current_section += line + " "

        # Add the last section
        if current_section:
            sections.append({
                'title': current_title or f"Section {len(sections) + 1}",
                'content': current_section.strip(),
                'page_number': page_number,
                'start_position': len(lines) - len(current_section.split('\n'))
            })

        # If no sections found, treat entire page as one section
        if not sections:
            sections.append({
                'title': f"Content from Page {page_number}",
                'content': text,
                'page_number': page_number,
                'start_position': 0
            })

        return sections

    def calculate_relevance_score(self, section_content: str, persona: str, job_description: str) -> float:
        """
        Calculate relevance score using multiple methods
        """
        # Combine persona and job for context
        context = f"{persona} {job_description}"

        # Method 1: Semantic similarity using sentence transformers
        try:
            section_embedding = self.sentence_model.encode([section_content])
            context_embedding = self.sentence_model.encode([context])
            semantic_score = cosine_similarity(section_embedding, context_embedding)[0][0]
        except:
            semantic_score = 0.0

        # Method 2: Keyword matching
        persona_keywords = self.extract_keywords(persona.lower())
        job_keywords = self.extract_keywords(job_description.lower())
        section_keywords = self.extract_keywords(section_content.lower())

        # Calculate keyword overlap
        all_target_keywords = persona_keywords.union(job_keywords)
        keyword_overlap = len(all_target_keywords.intersection(section_keywords))
        keyword_score = min(keyword_overlap / max(len(all_target_keywords), 1), 1.0)

        # Method 3: Content quality indicators
        quality_score = self.assess_content_quality(section_content)

        # Combine scores
        final_score = (semantic_score * 0.5) + (keyword_score * 0.3) + (quality_score * 0.2)

        return final_score

    def extract_keywords(self, text: str) -> set:
        """
        Extract meaningful keywords from text
        """
        # Tokenize and clean
        words = word_tokenize(text.lower())

        # Remove stopwords and short words
        keywords = set()
        for word in words:
            if (word not in self.stop_words and
                    len(word) > 2 and
                    word.isalpha()):
                keywords.add(self.lemmatizer.lemmatize(word))

        return keywords

    def assess_content_quality(self, content: str) -> float:
        """
        Assess the quality and informativeness of content
        """
        if not content or len(content) < 50:
            return 0.0

        # Factors that indicate quality content
        score = 0.0

        # Length (moderate length is good)
        length_score = min(len(content) / 1000, 1.0)  # Normalize to 1000 chars
        if length_score > 0.1:  # At least some content
            score += 0.3

        # Sentence structure
        sentences = sent_tokenize(content)
        if len(sentences) > 1:
            score += 0.2

        # Information density (presence of numbers, specific terms)
        if re.search(r'\d+', content):  # Contains numbers
            score += 0.1

        # Lists or structured content
        if re.search(r'[•\-\*]\s|^\d+\.', content, re.MULTILINE):
            score += 0.2

        # Presence of actionable content
        action_words = ['visit', 'try', 'explore', 'enjoy', 'experience', 'discover', 'learn']
        if any(word in content.lower() for word in action_words):
            score += 0.2

        return min(score, 1.0)

    def process_documents(self, input_config: Dict[str, Any], pdf_directory: str) -> Dict[str, Any]:
        """
        Main processing function
        """
        print("Starting document processing...")

        # Extract input parameters
        persona = input_config['persona']['role']
        job_description = input_config['job_to_be_done']['task']
        document_list = input_config['documents']

        print(f"Persona: {persona}")
        print(f"Job: {job_description}")
        print(f"Documents to process: {len(document_list)}")

        # Process each document
        all_sections = []
        processed_docs = []

        for doc_info in document_list:
            filename = doc_info['filename']
            pdf_path = os.path.join(pdf_directory, filename)

            if not os.path.exists(pdf_path):
                print(f"Warning: File {pdf_path} not found")
                continue

            print(f"Processing {filename}...")
            doc_data = self.extract_text_from_pdf(pdf_path)

            if doc_data:
                processed_docs.append(filename)

                # Process each page and section
                for page_data in doc_data['pages']:
                    for section in page_data['sections']:
                        # Calculate relevance score
                        relevance_score = self.calculate_relevance_score(
                            section['content'], persona, job_description
                        )

                        section_info = {
                            'document': filename,
                            'section_title': section['title'],
                            'page_number': section['page_number'],
                            'content': section['content'],
                            'relevance_score': relevance_score
                        }

                        all_sections.append(section_info)

        # Sort sections by relevance score
        all_sections.sort(key=lambda x: x['relevance_score'], reverse=True)

        # Select top sections for extracted_sections (top 5)
        top_sections = all_sections[:5]
        extracted_sections = []

        for i, section in enumerate(top_sections, 1):
            extracted_sections.append({
                'document': section['document'],
                'section_title': section['section_title'],
                'importance_rank': i,
                'page_number': section['page_number']
            })

        # Create subsection analysis (refined text for top sections)
        subsection_analysis = []
        for section in top_sections:
            # Refine the text (clean and optimize)
            refined_text = self.refine_text(section['content'])

            subsection_analysis.append({
                'document': section['document'],
                'refined_text': refined_text,
                'page_number': section['page_number']
            })

        # Create final output
        output = {
            'metadata': {
                'input_documents': processed_docs,
                'persona': persona,
                'job_to_be_done': job_description,
                'processing_timestamp': datetime.now().isoformat()
            },
            'extracted_sections': extracted_sections,
            'subsection_analysis': subsection_analysis
        }

        print(f"Processing complete! Found {len(all_sections)} sections, selected top {len(top_sections)}")
        return output

    def refine_text(self, text: str) -> str:
        """
        Clean and refine text for better readability
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
        text = re.sub(r'(\w)([;:])', r'\1\2 ', text)  # Space after punctuation

        # Ensure proper sentence endings
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)

        # Limit length if too long
        if len(text) > 1500:
            sentences = sent_tokenize(text)
            # Keep first few sentences that fit within limit
            refined = ""
            for sentence in sentences:
                if len(refined + sentence) < 1500:
                    refined += sentence + " "
                else:
                    break
            text = refined.strip()

        return text


def display_results(results):
    """Display the results nicely"""
    print(" RESULTS SUMMARY:")
    print("=" * 50)
    print(f"Persona: {results['metadata']['persona']}")
    print(f"Task: {results['metadata']['job_to_be_done']}")
    print(f"Documents processed: {len(results['metadata']['input_documents'])}")
    print(f"Processing time: {results['metadata']['processing_timestamp']}")

    print("\n TOP SECTIONS IDENTIFIED:")
    print("-" * 30)
    for i, section in enumerate(results['extracted_sections'], 1):
        print(f"{i}. {section['section_title']}")
        print(f"    Document: {section['document']}")
        print(f"   Page: {section['page_number']}")
        print()


def main():
    """
    Main function to run the document intelligence system
    """
    print("=== Document Intelligence System ===")
    current_dir = os.getcwd()
    pdf_directory = os.path.join(current_dir, "PDFs")
    input_file = os.path.join(current_dir, "challenge1b_input.json")
    output_file = os.path.join(current_dir, "challenge1b_output.json")

    print(f"Working directory: {current_dir}")
    print(f"PDF directory: {pdf_directory}")

    # Step 1: Setup NLTK data
    setup_nltk_data()
    if not check_pdf_files(pdf_directory):
        print("\n Please ensure PDF files are in the correct location before running.")
        return
    print("\nCreating input configuration...")
    input_config = create_input_config(input_file)
    print("\nInitializing system...")
    system = DocumentIntelligenceSystem()

    try:
        result = system.process_documents(input_config, pdf_directory)
        print("Saving results...")
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)

        print(f"Processing complete! Results saved to {output_file}")
        print("\n=== SUMMARY ===")
        print(f"Persona: {result['metadata']['persona']}")
        print(f"Job: {result['metadata']['job_to_be_done']}")
        print(f"Documents processed: {len(result['metadata']['input_documents'])}")
        print(f"Top sections identified: {len(result['extracted_sections'])}")

        print("\n=== TOP SECTIONS ===")
        for section in result['extracted_sections']:
            print(f"{section['importance_rank']}. {section['section_title']} "
                  f"(Page {section['page_number']}, {section['document']})")
        print("\n" + "=" * 50)
        display_results(result)

    except Exception as e:
        print(f" Error: {str(e)}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    main()