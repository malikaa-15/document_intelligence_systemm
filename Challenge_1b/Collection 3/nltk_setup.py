import nltk

# Download essential NLTK data
print("Downloading NLTK data...")

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

print("Downloads complete!")

# Test tokenization
from nltk.tokenize import word_tokenize
text = "Hello world! This is working."
tokens = word_tokenize(text)
print("Test successful:", tokens)