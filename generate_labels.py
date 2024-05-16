import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')  # Tokenizer model
nltk.download('stopwords')  # Stopwords list
stop_words = set(stopwords.words('english'))

def tokenize_documents(documents):
    tokenized = []
    for document in documents:
        words = word_tokenize(document)
        filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
        tokenized.append(filtered_words)
    return tokenized

def compute_metrics(query_set):
    results = {}
    
    for query, documents in query_set.items():
        vectorizer = CountVectorizer()
        doc_term_matrix = vectorizer.fit_transform(documents)
        query_vector = vectorizer.transform([query])
        
        # Compute TF statistics
        tf = np.array(doc_term_matrix.toarray())
        tf_query = np.array(query_vector.toarray())
        
        # Document length (stream length)
        doc_lengths = tf.sum(axis=1)
        
        # Covered query terms and ratios
        covered_query_terms = (tf_query > 0).sum()
        covered_query_term_ratio = covered_query_terms / len(vectorizer.get_feature_names_out())
        
        # Compute IDF
        tfidf_vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=False)
        tfidf_vectorizer.fit(documents + [query])
        idf = tfidf_vectorizer.idf_
        idf_dict = dict(zip(tfidf_vectorizer.get_feature_names_out(), idf))
        
        # Query terms IDF
        query_terms = vectorizer.get_feature_names_out()
        query_idfs = np.array([idf_dict[term] if term in idf_dict else 0 for term in query_terms])
        
        # TF-IDF calculations
        tfidf = tf * query_idfs
        
        # Stream length normalized TF
        norm_tf = tf / doc_lengths[:, np.newaxis]
        
        # Compute metrics for each document
        doc_metrics = []
        for i, doc in enumerate(documents):
            metrics = {
                "covered query term number": np.sum(tf_query > 0),
                "covered query term ratio": covered_query_term_ratio,
                "stream length": doc_lengths[i],
                "sum of term frequency": np.sum(tf[i]),
                "min of term frequency": np.min(tf[i]),
                "max of term frequency": np.max(tf[i]),
                "mean of term frequency": np.mean(tf[i]),
                "variance of term frequency": np.var(tf[i]),
                "sum of stream length normalized term frequency": np.sum(norm_tf[i]),
                "min of stream length normalized term frequency": np.min(norm_tf[i]),
                "max of stream length normalized term frequency": np.max(norm_tf[i]),
                "mean of stream length normalized term frequency": np.mean(norm_tf[i]),
                "variance of stream length normalized term frequency": np.var(norm_tf[i]),
                "sum of tf*idf": np.sum(tfidf[i]),
                "min of tf*idf": np.min(tfidf[i]),
                "max of tf*idf": np.max(tfidf[i]),
                "mean of tf*idf": np.mean(tfidf[i]),
                "variance of tf*idf": np.var(tfidf[i])
            }
            doc_metrics.append(metrics)
        
        # Compute BM25
        # Tokenize for BM25
        tokenized_documents = tokenize_documents(documents)
        tokenized_query = tokenize_documents([query])[0]  # Tokenize query similarly

        # Compute BM25
        bm25 = BM25Okapi(tokenized_documents)
        bm25_scores = bm25.get_scores(tokenized_query)
        
        
        # Add BM25 scores to metrics
        for i, score in enumerate(bm25_scores):
            doc_metrics[i]["BM25"] = score
        
        results[query] = doc_metrics

    return results

# Example Usage
# query_set = {
#     "What is AI?": ["Artificial Intelligence is the branch of engineering and science devoted to constructing machines that think.",
#                     "AI is the field of science which concerns itself with building hardware and software that replicates human functions such as learning and reasoning."],
#     "Explain machine learning": ["Machine learning is a type of artificial intelligence that enables self-learning from data and applies that learning without human intervention.",
#                                 "It is the scientific study of algorithms and statistical models that computer systems use to perform specific tasks."]
# }

# results = compute_metrics(query_set)
# print(results)
