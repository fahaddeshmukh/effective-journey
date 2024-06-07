import os
import torch
from torch_geometric.data import Data
import pandas as pd

def load_embeddings(embeddings_path):
    """
    Loads embeddings from a CSV file, excluding the last column (label).

    Args:
        embeddings_path (str): Path to the CSV file containing embeddings.

    Returns:
        tuple: A tuple containing the embeddings as a numpy array and the set of node IDs.
    """
    df = pd.read_csv(embeddings_path)
    embeddings = df.iloc[:, :-1].values  # Exclude the last column (label)
    node_ids = set(df.index)
    return embeddings, node_ids

def print_graph_info(graph_data, graph_name):
    """
    Prints detailed information about a PyTorch Geometric Data object.

    Args:
        graph_data (Data): A PyTorch Geometric Data object.
        graph_name (str): Name of the graph to be printed.
    """
    print(f"Graph: {graph_name}")
    print(f"Type of graph data: {type(graph_data)}")
    if isinstance(graph_data, Data):
        print(graph_data)
        print(f"Number of nodes: {graph_data.num_nodes}")
        print(f"Number of edges: {graph_data.num_edges}")
        print(f"Node feature shape: {graph_data.x.shape}")
        print(f"Edge index shape: {graph_data.edge_index.shape}")
    else:
        print(f"Graph data keys: {graph_data.keys()}")
        for key, value in graph_data.items():
            print(f"{key}: {type(value)}, {value.shape if hasattr(value, 'shape') else 'N/A'}")

def create_new_graph(original_graph, new_node_features, new_graph_path):
    """
    Creates a new graph with updated node features and saves it.

    Args:
        original_graph (Data): The original PyTorch Geometric Data object.
        new_node_features (numpy.ndarray): The new node features to be used.
        new_graph_path (str): Path to save the new graph.
    """
    # Clone the original graph
    new_graph = original_graph.clone()
    # Update the node features
    new_graph.x = torch.tensor(new_node_features, dtype=torch.float)
    # Save the new graph
    torch.save(new_graph, new_graph_path)
    print(f"New graph saved as {new_graph_path}")

if __name__ == "__main__":
    # Load the original graph
    original_graph_path = os.path.join(os.path.dirname(__file__), '../graphs/graph.pt')
    original_graph = torch.load(original_graph_path)
    
    # Print info about the original graph
    print_graph_info(original_graph, "Original Graph")

    # Load Word2Vec embeddings
    word2vec_embeddings_path = os.path.join(os.path.dirname(__file__), '../data/vec_sentence_embeddings.csv')
    word2vec_embeddings, word2vec_node_ids = load_embeddings(word2vec_embeddings_path)
    
    # Load RoBERTa embeddings
    roberta_embeddings_path = os.path.join(os.path.dirname(__file__), '../data/embeddings_with_labels.csv')
    roberta_embeddings, roberta_node_ids = load_embeddings(roberta_embeddings_path)
    
    # Check if the embeddings align with the graph
    expected_node_ids = set(range(original_graph.num_nodes))
    
    # Check alignment for Word2Vec
    mismatch_word2vec = expected_node_ids.symmetric_difference(word2vec_node_ids)
    if mismatch_word2vec:
        print("There is a mismatch in Word2Vec node IDs:", mismatch_word2vec)
    else:
        print("Word2Vec node IDs are aligned.")
        # Create and save Word2Vec graph
        word2vec_graph_path = os.path.join(os.path.dirname(__file__), '../graphs/wordvec_graph.pt')
        create_new_graph(original_graph, word2vec_embeddings, word2vec_graph_path)
        print_graph_info(torch.load(word2vec_graph_path), "Word2Vec Graph")

    # Check alignment for RoBERTa
    mismatch_roberta = expected_node_ids.symmetric_difference(roberta_node_ids)
    if mismatch_roberta:
        print("There is a mismatch in RoBERTa node IDs:", mismatch_roberta)
    else:
        print("RoBERTa node IDs are aligned.")
        # Create and save RoBERTa graph
        roberta_graph_path = os.path.join(os.path.dirname(__file__), '../graphs/roberta_graph.pt')
        create_new_graph(original_graph, roberta_embeddings, roberta_graph_path)
        print_graph_info(torch.load(roberta_graph_path), "RoBERTa Graph")
