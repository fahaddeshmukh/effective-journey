import pandas as pd
import re
import gensim
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
import os

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_data(file_path):
    """
    Preprocesses the data by combining title and abstract, removing URLs, 
    and performing tokenization and stopword removal.

    Args:
        file_path (str): Path to the CSV file containing paper information.

    Returns:
        pd.DataFrame: DataFrame with preprocessed text.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    print(f"Original DataFrame length: {len(df)}")

    df['combined_text'] = df['title'] + ": " + df['abstract'].fillna('')

    def extract_urls(text):
        url_pattern = r'https?://\S+'
        urls = re.findall(url_pattern, text)
        return urls

    all_urls = []
    for text in df['combined_text']:
        urls = extract_urls(text)
        all_urls.extend(urls)

    num_urls = len(all_urls)
    print(f"Total number of URLs found in abstracts: {num_urls}")

    def remove_urls(text):
        url_pattern = r'https?://\S+'
        clean_text = re.sub(url_pattern, '', text)
        return clean_text

    df['combined_text'] = df['combined_text'].apply(remove_urls)

    all_urls = []
    for text in df['combined_text']:
        urls = extract_urls(text)
        all_urls.extend(urls)

    num_urls = len(all_urls)
    print(f"Total number of URLs found in abstracts after cleaning: {num_urls}")

    df['processed_text'] = df['combined_text'].apply(gensim.utils.simple_preprocess)

    def remove_stopwords(tokens):
        return [word for word in tokens if word not in stop_words]

    df['processed_text'] = df['processed_text'].apply(remove_stopwords)

    print(f"Processed DataFrame length: {len(df)}")
    
    return df

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), '../data/paper_info.csv')
    processed_path = os.path.join(os.path.dirname(__file__), '../data/processed_data.csv')
    
    df = preprocess_data(data_path)
    df.to_csv(processed_path, index=False)
    print(f"Saved processed data to {processed_path}")
