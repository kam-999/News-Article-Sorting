FROM continuumio/miniconda3

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y iputils-ping && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/nltk_data && chmod -R 777 /app/nltk_data

RUN python -m nltk.downloader -d /app/nltk_data stopwords wordnet omw-1.4 averaged_perceptron_tagger_eng

ENV NLTK_DATA=/app/nltk_data

CMD ["python3", "app.py"]