# Use a base Python image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy entire project into the container
COPY . .

# Install Python packages from Collection 3 (superset of all)
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r "Collection 3/requirements.txt"

# Download NLTK and spaCy models
RUN python -m spacy download en_core_web_sm
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Set the default command to accept a collection folder name
ENTRYPOINT ["python"]
CMD ["Collection 3/main.py"]
