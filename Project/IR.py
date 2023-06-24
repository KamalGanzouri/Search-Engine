import string
from typing import Any

import numpy as np
import pandas as pd
import regex as re
from nltk import word_tokenize
from nltk.corpus import stopwords
from numpy.linalg import norm

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
stopwords = set(stopwords.words('english'))
DOCS: dict[Any, Any] = {}
row_DOCS: dict[Any, Any] = {}
POSITIONAL_INDEX = {}


def read_files():
    # Read 10 files (.txt)
    for i in range(1, 11):
        # Read every single file
        DOC = open("E:/Collage/IR/DocumentCollection/" + str(i) + ".txt", 'r').read()
        row_DOCS["DOC" + str(i)] = DOC
        # Apply tokenization
        DOC = apply_tokenization(DOC)
        # Apply Stop words (except [in,to])
        DOC = apply_Stopwords(DOC)
        DOCS["DOC" + str(i)] = DOC


def apply_tokenization(DOC):
    DOC = re.sub('[%s]' % re.escape(string.punctuation), ' ', DOC)
    DOC = re.sub('\w*\d\w*', ' ', DOC).lower()
    DOC = word_tokenize(DOC)
    return DOC


def apply_Stopwords(DOC):
    Final_DOC = []
    for w in DOC:
        if w == "in" or w == "to" or w == "where":
            Final_DOC.append(w)
        elif w not in stopwords:
            Final_DOC.append(w)
    return Final_DOC


def positional_index():
    for i in DOCS:
        for pos, term in enumerate(DOCS[i]):
            # If term already exists in the positional index dictionary.
            if term in POSITIONAL_INDEX:

                # Increment total freq by 1.
                POSITIONAL_INDEX[term][0] = POSITIONAL_INDEX[term][0] + 1

                # Check if the term has existed in that DocID before.
                if i in POSITIONAL_INDEX[term][1]:
                    POSITIONAL_INDEX[term][1][i].append(pos)

                else:
                    POSITIONAL_INDEX[term][1][i] = [pos]
            else:

                # Initialize the list.
                POSITIONAL_INDEX[term] = []
                # The total frequency is 1.
                POSITIONAL_INDEX[term].append(1)
                # The postings list is initially empty.
                POSITIONAL_INDEX[term].append({})
                # Add doc ID to postings list.
                POSITIONAL_INDEX[term][1][i] = [pos]


def ComputeTF():
    tf = pd.DataFrame(np.zeros((len(POSITIONAL_INDEX.keys()), len(DOCS.keys()))), columns=list(DOCS.keys()),
                      index=list(POSITIONAL_INDEX.keys()))
    for w in POSITIONAL_INDEX.keys():
        for d in POSITIONAL_INDEX[w][1].keys():
            tf[d][w] = tf[d][w] + 1
    return tf


def ComputeWTF():
    tf = ComputeTF()
    wtf = pd.DataFrame(np.zeros((len(POSITIONAL_INDEX.keys()), len(DOCS.keys()))), columns=list(DOCS.keys()),
                       index=list(POSITIONAL_INDEX.keys()))
    for w in POSITIONAL_INDEX.keys():
        for d in POSITIONAL_INDEX[w][1].keys():
            wtf[d][w] = wtf[d][w] + 1 + np.log10(tf[d][w])
    return wtf


def ComputeIDF():
    idf = pd.DataFrame(np.zeros((len(POSITIONAL_INDEX.keys()), 2)), columns=['df', 'idf'],
                       index=list(POSITIONAL_INDEX.keys()))
    for w in POSITIONAL_INDEX.keys():
        idf['df'][w] = POSITIONAL_INDEX[w][0]
        idf['idf'][w] = np.log10(len(DOCS) / POSITIONAL_INDEX[w][0])
    return idf


def ComputeTF_IDF():
    tf_idf = pd.DataFrame(np.zeros((len(POSITIONAL_INDEX.keys()), len(DOCS.keys()))), columns=list(DOCS.keys()),
                          index=list(POSITIONAL_INDEX.keys()))
    wtf = ComputeWTF()
    idf = ComputeIDF()['idf']
    for w in POSITIONAL_INDEX.keys():
        for i in DOCS:
            tf_idf[i][w] = wtf[i][w] * idf[w]
    return tf_idf


def ComputeDOC_LEN():
    doc_len = pd.DataFrame(np.zeros((len(DOCS.keys()), 1)), columns=['Length'], index=list(DOCS.keys()))
    tf_idf = ComputeTF_IDF()
    for d in DOCS:
        total = 0
        for values in tf_idf[d].tolist():
            total += np.square(values)
        doc_len['Length'][d] = np.sqrt(total)
    return doc_len


def ComputeNORM_TF_IDF():
    norm_tf_idf = ComputeTF_IDF()
    doc_len = ComputeDOC_LEN()
    for w in POSITIONAL_INDEX.keys():
        for i in DOCS:
            norm_tf_idf[i][w] = norm_tf_idf[i][w] / doc_len['Length'][i]
    return norm_tf_idf


def ComputeQUERY(QUERY):
    QUERY = apply_tokenization(QUERY)
    QUERY = apply_Stopwords(QUERY)
    query_compute = pd.DataFrame(np.zeros((len(list(set(QUERY))), 5)),
                                 columns=['TF', 'WTF', 'IDF', 'TF-IDF', 'NORMALIZED'],
                                 index=list(set(QUERY)))

    for w in QUERY:
        query_compute['TF'][w] = query_compute['TF'][w] + 1
        query_compute['WTF'][w] = query_compute['WTF'][w] + 1 + np.log10(query_compute['TF'][w])
        query_compute['IDF'][w] = ComputeIDF()['idf'][w]
        query_compute['TF-IDF'][w] = query_compute['WTF'][w] * query_compute['IDF'][w]
    total = 0
    for values in query_compute['TF-IDF'].tolist():
        total += np.square(values)
    query_len = np.sqrt(total)
    for w in QUERY:
        query_compute['NORMALIZED'][w] = query_compute['TF-IDF'][w] / query_len
    return query_compute, query_len


def NormalizationDoc(nor_query):
    MATCH = search_query_all(nor_query)
    QUERY = apply_tokenization(nor_query)
    QUERY = apply_Stopwords(QUERY)
    computedquery, querylength = ComputeQUERY(UserQuery)

    norm_matched = pd.DataFrame(np.zeros((len(MATCH) + 1, len(QUERY))), columns=sorted(list(MATCH)),
                                index=QUERY + ['sum'])
    for d in MATCH:
        for w in QUERY:
            product = ComputeNORM_TF_IDF()[d][w] * computedquery['NORMALIZED'][w]
            norm_matched[d][w] = product
            norm_matched[d]['sum'] += product
    return norm_matched


def search_word(word):
    if word in POSITIONAL_INDEX:
        MATCH = set()
        for keys, values in POSITIONAL_INDEX[word][1].items():
            MATCH.add(str(keys))
        return MATCH


def search_query_all(enter_query):
    MATCH = set()
    QUERY = apply_tokenization(enter_query)
    QUERY = apply_Stopwords(QUERY)
    if not QUERY:
        return None
    if search_word(QUERY[0]) is None:
        return None
    else:
        MATCH.update(search_word(QUERY[0]))
    for i in QUERY:
        if search_word(i) is None:
            return None
        else:
            MATCH = MATCH.intersection(search_word(i))
    if MATCH == set():
        return None

    return MATCH


def Cosine(cosine_query):
    List = []
    if search_query_all(cosine_query) is None:
        return None
    List.extend(search_query_all(cosine_query))
    List.append("Query")
    QUERY = apply_tokenization(cosine_query)
    QUERY = apply_Stopwords(QUERY)

    df = pd.DataFrame(np.zeros((len(POSITIONAL_INDEX.keys()), len(List))), columns=List,
                      index=list(POSITIONAL_INDEX.keys()))
    for i in List:  # Words in the Match
        if i == "Query":
            for x in QUERY:
                df[i][x] = df[i][x] + 1
            continue
        for w in DOCS[i]:
            df[i][w] = df[i][w] + 1
    return df


def CosineSimilarity(similarity_query):
    if Cosine(similarity_query) is None:
        return None
    df = Cosine(similarity_query)
    CosineDict = {}
    A = df["Query"].tolist()
    for i in df.keys():
        if i == "Query":
            continue
        B = df.loc[:, i].tolist()
        CosineDict[i] = np.dot(A, B) / (norm(A) * norm(B))
    return CosineDict


def RankWithSimilarity(rank_query):
    if CosineSimilarity(rank_query) is None:
        return None
    lis = CosineSimilarity(rank_query)
    lis = sorted(lis.items(), key=lambda x: x[1], reverse=True)
    for item in lis:
        print(item[0])


# Read File And Apply tokenization Apply stopwords (except [in,to])
read_files()
# Build positional index
print("---------- Display positional index ----------\n \n")
positional_index()
# Display positional index
for words in POSITIONAL_INDEX:
    print(words + " : " + str(POSITIONAL_INDEX[words]))
# Compute term frequency for each term in each document.
TF = ComputeTF()
WF = ComputeWTF()
# Display Compute term frequency for each term in each document.
print("\n \n---------- Display Compute term frequency for each term in each document ----------\n \n")
print(TF)
print("\n \n---------- Display Compute wighted term frequency for each term in each document ----------\n \n")
print(WF)
# Compute IDF for each term.
IDF = ComputeIDF()
# Display IDF for each term.
print("\n \n---------- Display IDF for each term.----------\n \n")
print(IDF)
# Displays TF.IDF matrix.
print("\n \n---------- Displays TF.IDF matrix ----------\n \n")
print(ComputeTF_IDF())
print("\n \n---------- Displays DOCUMENT Length ----------\n \n")
print(ComputeDOC_LEN())
print("\n \n---------- Displays NORMALIZED TF.IDF matrix ----------\n \n")
print(ComputeNORM_TF_IDF())

while True:
    # Allow users to write phrase query
    print("---------- Allow users to write phrase query ----------\n \n")
    UserQuery = input("Type the Query: ")
    if UserQuery == "0":
        break
    # System returns the matched documents for the query
    print("\n \n---------- System returns the matched documents for the query ---------- \n \n")
    result = search_query_all(UserQuery)
    for key in sorted(result):
        print(key + ": " + row_DOCS[key])
        print("\n")
    computed_query, query_length = ComputeQUERY(UserQuery)
    print("---------- System returns computes for the query ---------- \n \n")
    print(computed_query)
    print("\nQuery Length : " + str(query_length))
    print("\n \n---------- System returns Normalization Doc similarity  for the query ---------- \n \n")
    NS = NormalizationDoc(UserQuery)
    print(NS)
    print("\n \n---------- Display similarity between the query and matched documents----------\n \n")
    for key, value in sorted(NS.items()):
        print("Cosine Similarity for " + str(key) + " equals " + str(value['sum']) + "\n")
    # Compute cosine similarity between the query and matched documents.
    CQ = CosineSimilarity(UserQuery)
    # Display cosine similarity between the query and matched documents.
    print("\n \n---------- Display cosine similarity between the query and matched documents----------\n \n")
    for key, value in sorted(CQ.items()):
        print("Cosine Similarity for " + key + " equals " + str(value) + "\n")
    # Rank documents based on cosine similarity.
    print("\n---------- Rank documents based on similarity----------\n \n")
    RankWithSimilarity(UserQuery)
    print("\n")
    print("______________________________________________________________________________________________________\n\n")
