import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

def plot_tsne(data_file, output_file_2d, output_file_3d):
    """
    Plots t-SNE visualizations of embeddings in 2D and 3D.

    Args:
        data_file (str): Path to the CSV file containing embeddings and labels.
        output_file_2d (str): Path to save the 2D t-SNE plot.
        output_file_3d (str): Path to save the 3D t-SNE plot.

    Returns:
        None
    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"File not found: {data_file}")

    df = pd.read_csv(data_file)

    if 'label' not in df.columns:
        raise KeyError(f"The DataFrame from {data_file} does not contain a 'label' column.")

    top_labels = df['label'].value_counts().head(5).index
    filtered_df = df[df['label'].isin(top_labels)]
    sampled_df = filtered_df.sample(n=5000, random_state=42)
    embeddings = sampled_df.iloc[:, :-1].values  # Assuming embeddings are all columns except the last one
    labels = sampled_df['label'].values

    # 2D t-SNE
    tsne_2d = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
    print("Computing 2D t-SNE...")
    reduced_embeddings_2d = tsne_2d.fit_transform(embeddings)

    reduced_df_2d = pd.DataFrame(reduced_embeddings_2d, columns=['Dim1', 'Dim2'])
    reduced_df_2d['label'] = labels

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Dim1', y='Dim2', hue='label', palette='viridis', data=reduced_df_2d, s=50, alpha=0.7)
    plt.title('t-SNE Visualization of Embeddings in 2D')
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.savefig(output_file_2d)
    plt.show()

    # 3D t-SNE
    tsne_3d = TSNE(n_components=3, perplexity=30, n_iter=300, random_state=42)
    print("Computing 3D t-SNE...")
    reduced_embeddings_3d = tsne_3d.fit_transform(embeddings)

    reduced_df_3d = pd.DataFrame(reduced_embeddings_3d, columns=['Dim1', 'Dim2', 'Dim3'])
    reduced_df_3d['label'] = labels

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(reduced_df_3d['Dim1'], reduced_df_3d['Dim2'], reduced_df_3d['Dim3'], c=reduced_df_3d['label'].astype('category').cat.codes, cmap='viridis')
    legend1 = ax.legend(*scatter.legend_elements(), title="Labels")
    ax.add_artist(legend1)
    plt.title('t-SNE Visualization of Embeddings in 3D')
    plt.savefig(output_file_3d)
    plt.show()

if __name__ == "__main__":
    vec_embeddings_path = os.path.join(os.path.dirname(__file__), '../data/vec_sentence_embeddings.csv')
    roberta_embeddings_path = os.path.join(os.path.dirname(__file__), '../data/embeddings_with_labels.csv')
    
    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print(f"Plotting t-SNE for Word2Vec embeddings from: {vec_embeddings_path}")
    plot_tsne(vec_embeddings_path, os.path.join(results_dir, 'tsne_word2vec_2d.png'), os.path.join(results_dir, 'tsne_word2vec_3d.png'))

    print(f"Plotting t-SNE for RoBERTa embeddings from: {roberta_embeddings_path}")
    plot_tsne(roberta_embeddings_path, os.path.join(results_dir, 'tsne_roberta_2d.png'), os.path.join(results_dir, 'tsne_roberta_3d.png'))
