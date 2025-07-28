import json
import os
import re
from datetime import datetime
from typing import List, Dict
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentIntelligenceSystem:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.8,
            min_df=0.1
        )

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        try:
            doc = fitz.open(pdf_path)
            page_texts = {}
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    page_texts[page_num + 1] = text
            doc.close()
            return page_texts
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return {}

    def preprocess_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)]', ' ', text)
        return text.lower()

    def extract_sections(self, text: str, page_num: int) -> List[Dict]:
        sections = []
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        current_section = None
        current_content = []

        for para in paragraphs:
            if self.is_likely_header(para):
                if current_section and current_content:
                    sections.append({
                        'title': current_section,
                        'content': ' '.join(current_content),
                        'page_number': page_num
                    })
                current_section = para[:100]
                current_content = []
            else:
                if current_section:
                    current_content.append(para)
                else:
                    if not current_section:
                        current_section = f"Content from page {page_num}"
                    current_content.append(para)

        if current_section and current_content:
            sections.append({
                'title': current_section,
                'content': ' '.join(current_content),
                'page_number': page_num
            })

        return sections

    def is_likely_header(self, text: str) -> bool:
        text = text.strip()
        if len(text) > 200 or len(text) < 5:
            return False
        header_patterns = [
            r'^[A-Z][a-z].*[^.!?]$',
            r'^\d+\.\s*[A-Z]',
            r'^[A-Z\s]+$',
            r'.\s(tool|function|feature|step|method|process)s?\s$',
        ]
        for pattern in header_patterns:
            if re.match(pattern, text):
                return True
        if text.endswith('?'):
            return True
        action_words = ['create', 'convert', 'edit', 'export', 'fill', 'sign', 'share', 'request']
        first_word = text.split()[0].lower() if text.split() else ''
        return first_word in action_words

    def calculate_relevance_score(self, section_content: str, persona: str, job_description: str) -> float:
        target_text = f"{persona} {job_description}".lower()
        section_text = section_content.lower()
        target_keywords = self.extract_keywords(target_text)
        section_keywords = self.extract_keywords(section_text)
        if not target_keywords or not section_keywords:
            keyword_score = 0
        else:
            overlap = len(set(target_keywords) & set(section_keywords))
            keyword_score = overlap / max(len(target_keywords), len(section_keywords))

        try:
            all_texts = [target_text, section_text]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            similarity = 0

        final_score = 0.4 * keyword_score + 0.6 * similarity
        if 'form' in section_text or 'fill' in section_text or 'sign' in section_text:
            final_score += 0.2
        return min(final_score, 1.0)

    def extract_keywords(self, text: str) -> List[str]:
        words = word_tokenize(text.lower())
        return [
            self.stemmer.stem(word) for word in words
            if word not in self.stop_words and len(word) > 2 and word.isalpha()
        ]

    def refine_text_for_subsection(self, content: str, max_length: int = 300) -> str:
        sentences = sent_tokenize(content)
        refined_text = ""
        current_length = 0
        for sentence in sentences:
            if current_length + len(sentence) <= max_length:
                refined_text += sentence + " "
                current_length += len(sentence)
            else:
                break
        return refined_text.strip()

    def process_documents(self, input_data: Dict) -> Dict:
        logger.info("Starting document processing...")
        start_time = datetime.now()
        documents = input_data['documents']
        persona_role = input_data['persona']['role']
        job_task = input_data['job_to_be_done']['task']
        logger.info(f"Processing {len(documents)} documents for {persona_role}")
        all_sections = []

        for doc_info in documents:
            filename = doc_info['filename']
            filepath = os.path.join('PDFs', filename)
            if not os.path.exists(filepath):
                logger.warning(f"File not found: {filepath}")
                continue
            logger.info(f"Processing {filename}...")
            page_texts = self.extract_text_from_pdf(filepath)

            for page_num, text in page_texts.items():
                processed_text = self.preprocess_text(text)
                sections = self.extract_sections(processed_text, page_num)
                for section in sections:
                    relevance_score = self.calculate_relevance_score(
                        section['content'], persona_role, job_task
                    )
                    section.update({
                        'document': filename,
                        'relevance_score': relevance_score
                    })
                    all_sections.append(section)

        all_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
        top_sections = all_sections[:5]
        extracted_sections = []
        subsection_analysis = []

        for i, section in enumerate(top_sections):
            extracted_sections.append({
                'document': section['document'],
                'section_title': section['title'],
                'importance_rank': i + 1,
                'page_number': section['page_number']
            })
            refined_text = self.refine_text_for_subsection(section['content'])
            subsection_analysis.append({
                'document': section['document'],
                'refined_text': refined_text,
                'page_number': section['page_number']
            })

        output = {
            'metadata': {
                'input_documents': [doc['filename'] for doc in documents],
                'persona': persona_role,
                'job_to_be_done': job_task,
                'processing_timestamp': start_time.isoformat()
            },
            'extracted_sections': extracted_sections,
            'subsection_analysis': subsection_analysis
        }

        logger.info(f"Processing completed in {(datetime.now() - start_time).total_seconds():.2f} seconds")
        return output

def main():
    system = DocumentIntelligenceSystem()


    input_path = 'challenge1b_input.json'

    if not os.path.exists(input_path):
        print(f" Input file not found: {input_path}")
        return

    with open(input_path, 'r') as f:
        input_data = json.load(f)

    result = system.process_documents(input_data)

    with open('challenge1b_output.json', 'w') as f:
        json.dump(result, f, indent=4)

    print("Processing completed! Output saved to challenge1b_output.json")
    print(f" Found {len(result['extracted_sections'])} relevant sections")

if __name__ == "__main__":
    main()
