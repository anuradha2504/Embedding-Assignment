Text Embedding Techniques - NLP Assignment
This project explores various text embedding techniques using a sample dataset of product reviews. The goal is to understand how different embedding models represent textual data and to compare their effectiveness for downstream tasks.

📁 Dataset
The dataset used contains customer product reviews in a CSV format with the following fields:

review: Raw user review text.

cleaned_review: Preprocessed version of the review (lowercased, tokenized, lemmatized, etc.)

🔧 Preprocessing
Before applying embedding techniques, the text undergoes:

Lowercasing

Removal of punctuation and numbers

Tokenization

Stopword removal

Lemmatization

Removal of extra spaces

🧰 Libraries Used
pandas, numpy — Data handling

nltk, re — Text preprocessing

sklearn — BoW, TF-IDF, PCA

gensim — Word2Vec, FastText

matplotlib, seaborn — Visualization

📚 Embedding Techniques Implemented
Technique	Description
Bag of Words	Converts text to word count matrix using CountVectorizer.
TF-IDF	Converts text to weighted frequency matrix using TfidfVectorizer.
Word2Vec	Trains embedding using Skip-gram or CBOW on tokenized reviews.
GloVe	Loads pre-trained GloVe embeddings and maps them to review vectors.
FastText	Trains embeddings using subword information (n-grams) with Gensim.

🧪 Analysis Performed
Cosine similarity checks between words

OOV (Out-of-Vocabulary) behavior comparison

PCA-based visualization of embeddings

Word similarity search across models

Qualitative comparison of techniques (context-awareness, handling misspellings, etc.)

📊 Visualization Samples
Word embeddings visualized using PCA

Similar words for terms like cheap, love, bad, etc.

Review-level average embeddings

🚀 How to Run
Open in Google Colab or Jupyter Notebook

Run the preprocessing cells to clean the dataset

Execute each embedding section one-by-one:

Bag of Words

TF-IDF

Word2Vec

GloVe

FastText

Review visualizations and comparisons

(Optional) Extend by applying embeddings to a classifier (e.g., sentiment analysis)

📌 Learning Outcomes
Understand the principles behind embedding models

Learn preprocessing steps required for text embeddings

Compare context-aware and context-unaware models

Visualize word relationships in vector space

📎 References
Gensim Documentation

Stanford GloVe

scikit-learn feature extraction

Author:
Anuradha Kumari
