import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from tqdm import tqdm
import ast
import os

def train_word2vec(df):
    """
    Trains a Word2Vec model and generates TF-IDF weighted sentence embeddings.

    Args:
        df (pd.DataFrame): DataFrame with preprocessed text and labels.

    Returns:
        None
    """
    # Convert the 'processed_text' column from string representation of lists to actual lists
    df['processed_text'] = df['processed_text'].apply(ast.literal_eval)
    
    tokenized_text = df['processed_text'].tolist()
    text_as_strings = [' '.join(tokens) for tokens in tokenized_text]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(text_as_strings)
    tfidf_matrix = vectorizer.transform(text_as_strings)

    # Train Word2Vec model
    model = Word2Vec(vector_size=300, window=7, min_count=3, workers=4, sg=0)
    model.build_vocab(tokenized_text, progress_per=1000)
    model.train(tokenized_text, total_examples=model.corpus_count, epochs=model.epochs)
    
    # Ensure the models directory exists
    models_dir = os.path.join(os.path.dirname(__file__), '../models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    model.save(os.path.join(models_dir, "word2vec_model.model"))
    print("Word2Vec model has been trained and saved as '../models/word2vec_model.model'.")

    def get_weighted_average(sentence_tokens, model, tfidf_vector, vectorizer):
        word_vecs = []
        weights = []

        for word in sentence_tokens:
            if word in model.wv:
                word_vec = model.wv[word]
                word_vecs.append(word_vec)
                try:
                    word_idx = vectorizer.vocabulary_[word]
                    tfidf_weight = tfidf_vector[0, word_idx]
                except KeyError:
                    tfidf_weight = 0.0
                weights.append(tfidf_weight)

        if len(word_vecs) > 0:
            word_vecs = np.array(word_vecs)
            weights = np.array(weights)
            return np.average(word_vecs, axis=0, weights=weights)
        else:
            return np.zeros(model.vector_size)

    vec_embeddings = []
    for i, sentence_tokens in tqdm(enumerate(tokenized_text), total=len(tokenized_text), desc="Generating embeddings"):
        tfidf_vector = tfidf_matrix[i]
        weighted_avg_vec = get_weighted_average(sentence_tokens, model, tfidf_vector, vectorizer)
        vec_embeddings.append(weighted_avg_vec)

    vec_embeddings_df = pd.DataFrame(vec_embeddings)
    vec_embeddings_df['label'] = df['label']
    vec_embeddings_df.to_csv(os.path.join(os.path.dirname(__file__), '../data/vec_sentence_embeddings.csv'), index=False)
    print("Embeddings have been saved to '../data/vec_sentence_embeddings.csv'.")

if __name__ == "__main__":
    processed_path = os.path.join(os.path.dirname(__file__), '../data/processed_data.csv')
    
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"File not found: {processed_path}")
    
    df = pd.read_csv(processed_path)
    train_word2vec(df)
