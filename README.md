```markdown
# Search Engine

This project is a Search Engine implemented in Python. It reads a collection of text documents, processes them, and allows users to perform various search queries on the documents. The search engine utilizes techniques such as tokenization, removal of stopwords, positional indexing, term frequency (TF), inverse document frequency (IDF), and cosine similarity to rank the relevance of documents to the search queries.

## Features

- **Read and Process Documents**: The search engine reads a collection of text documents and processes them by tokenizing the text and removing stopwords.
- **Positional Indexing**: Builds a positional index for terms in the documents.
- **Term Frequency (TF)**: Computes the term frequency for each term in each document.
- **Weighted Term Frequency (WTF)**: Computes the weighted term frequency for each term in each document.
- **Inverse Document Frequency (IDF)**: Computes the inverse document frequency for each term.
- **TF-IDF**: Computes the TF-IDF matrix for the documents.
- **Cosine Similarity**: Computes the cosine similarity between the query and the documents to rank the documents based on their relevance.

## Installation

To run this project, you need to have Python installed on your system. Additionally, you need to install the required libraries using the following command:

```bash
pip install numpy pandas regex nltk
```

## Usage

1. **Prepare the Document Collection**: Place your text documents in the directory specified in the `read_files` function (e.g., `E:/Collage/IR/DocumentCollection/`).

2. **Run the Search Engine**: Execute the `IR.py` script to start the search engine.

3. **Perform Search Queries**: When prompted, enter your search query. The search engine will return the matched documents, compute various metrics, and display the results along with the cosine similarity and ranking of the documents.

## Code Overview

- **read_files**: Reads and processes the text documents.
- **apply_tokenization**: Tokenizes the text by removing punctuation and digits and converting it to lowercase.
- **apply_Stopwords**: Removes stopwords from the tokenized text.
- **positional_index**: Builds a positional index for the terms in the documents.
- **ComputeTF**: Computes the term frequency for each term in each document.
- **ComputeWTF**: Computes the weighted term frequency for each term in each document.
- **ComputeIDF**: Computes the inverse document frequency for each term.
- **ComputeTF_IDF**: Computes the TF-IDF matrix for the documents.
- **ComputeDOC_LEN**: Computes the length of each document.
- **ComputeNORM_TF_IDF**: Computes the normalized TF-IDF matrix for the documents.
- **ComputeQUERY**: Processes the user query and computes various metrics for the query.
- **NormalizationDoc**: Normalizes the query and computes the similarity between the query and the documents.
- **search_word**: Searches for a single term in the positional index.
- **search_query_all**: Searches for all terms in the query in the positional index.
- **Cosine**: Computes the cosine similarity matrix for the query.
- **CosineSimilarity**: Computes the cosine similarity between the query and the documents.
- **RankWithSimilarity**: Ranks the documents based on cosine similarity.
