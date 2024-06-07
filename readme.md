# Text Embeddings

This repository contains all the files and scripts required to generate text embeddings using two different techniques: 
1. Word2Vec
2. All-DistilRoBERTa-v1 Sentence Transformer

## Repository Structure

text-embeddings/
├── main.py
├── bin/
│   ├── data_util.py
│   ├── word2vec.py
│   ├── sentence_transformer.py
│   ├── results.py
│   ├── graph_builder.py
├── data/
│   ├── paper_info.csv
│   ├── processed_data.csv
├── graphs/
│   ├── graph.pt
│   ├── roberta_graph.pt
│   ├── word2vec_graph.pt
├── results/
│   ├── tsne_word2vec_2d.png
│   ├── tsne_word2vec_3d.png
│   ├── tsne_roberta_2d.png
│   ├── tsne_roberta_3d.png



## Project Overview

This project aims to generate text embeddings using both Word2Vec and a fine-tuned Sentence Transformer model (`all-distilroberta-v1`). It consists of multiple steps to preprocess data, train models, generate embeddings, and visualize them. Additionally, it creates graphs using the generated embeddings for further analysis.

### Steps to Execute the Project

1. **Place Initial Data**:
   - Place the initial `paper_info.csv` file into the `data/` folder.

2. **Run Main Script**:
   - Execute the following command from the command line:
     ```
     python main.py
     ```
   - This script will automatically execute the following steps:

### Step-by-Step Execution

1. **Preprocess Data**:
   - The script `data_util.py` preprocesses the given textual data by removing URLs, punctuation, stopwords, etc.
   - Processed data is saved as `processed_data.csv` in the `data/` directory.

2. **Train Word2Vec Model**:
   - The script `word2vec.py` trains a Word2Vec model using the processed text and generates embeddings for the sentences using TF-IDF weighted average.
   - Embeddings and model files are saved in the `data/` and `models/` directories, respectively.

3. **Fine-tune Sentence Transformer**:
   - The script `sentence_transformer.py` generates triplet data [Anchor, Positive, Negative] based on class labels.
   - A pre-trained Sentence Transformer (`all-distilroberta-v1`) is fine-tuned on this dataset, and embeddings are generated using this model.
   - The fine-tuned model and embeddings are saved in the respective directories.
   - **Interactive Training Metrics**: View the interactive training metrics for the fine-tuning process [here](https://wandb.ai/fahaddeshmukh1/fine-tune-embeddings/reports/Fine-tuning-metrics-for-all-distilroberta-v1-sentence-transformer---Vmlldzo4MjUwOTE5?accessToken=a9w8mmkb183avfz3habwgiars7y100i7b3nzh30mufcniv47sv5tcgqz4j5vs7s0).

4. **Visualize Embeddings**:
   - The script `results.py` visualizes the embeddings using t-SNE for dimensionality reduction into 2D and 3D plots.
   - The plots are saved in the `results/` directory.

5. **Create Graphs**:
   - Navigate to the `bin/` directory and run the following command:
     ```
     python graph_builder.py
     ```
   - This script creates graphs with the newly generated embeddings as node features.
   - The graphs are saved in the `graphs/` directory and key features of each graph are displayed.

## Interpretation of t-SNE Plots

Although 300 dimensions from Word2Vec and 768 from RoBERTa have been reduced to 2D/3D using t-SNE, the plots show separation of clusters (not perfect) among the embeddings based on class labels. The embeddings formed from Word2Vec + TF-IDF weighted average exhibit slightly better separation among classes compared to the RoBERTa embeddings.

## Requirements

- Python 3.x
- Required Python libraries (install using `requirements.txt` if provided or install manually as required):
  - `pandas`
  - `nltk`
  - `gensim`
  - `torch`
  - `sentence-transformers`
  - `sklearn`
  - `matplotlib`
  - `tqdm`

## How to Use

1. **Preprocess Data**:
   - Place `paper_info.csv` in the `data/` directory.
   - From the root run `main.py`:
     ```
     python main.py
     ```

2. **Generate Graphs**:
   - Navigate to the `bin/` directory.
   - Run the `graph_builder.py` script:
     ```
     python graph_builder.py
     ```


## Acknowledgments

- [NLTK](https://www.nltk.org/)
- [Gensim](https://radimrehurek.com/gensim/)
- [Hugging Face's Transformers](https://huggingface.co/transformers/)
- [Hugging Face Blog: Train Sentence Transformers](https://huggingface.co/blog/train-sentence-transformers)
- **Interactive Training Metrics**: [View report on W&B](https://wandb.ai/fahaddeshmukh1/fine-tune-embeddings/reports/Fine-tuning-metrics-for-all-distilroberta-v1-sentence-transformer---Vmlldzo4MjUwOTE5?accessToken=a9w8mmkb183avfz3habwgiars7y100i7b3nzh30mufcniv47sv5tcgqz4j5vs7s0)
