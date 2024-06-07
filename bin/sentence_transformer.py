import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
from itertools import combinations, cycle
from tqdm import tqdm
import numpy as np
import os
import wandb
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

def generate_triplet_data(df, samples_per_class=500):
    """
    Generates triplet data for training the SentenceTransformer model.

    Args:
        df (pd.DataFrame): DataFrame with preprocessed text and labels.
        samples_per_class (int): Number of samples per class to generate.

    Returns:
        triplet_df (pd.DataFrame): DataFrame containing triplet data.
    """
    triplets = []
    unique_labels = df['label'].unique()
    label_cycle = cycle(unique_labels)

    for label in tqdm(unique_labels, desc="Generating triplets"):
        current_group = df[df['label'] == label]
        other_groups = df[df['label'] != label]
        if len(current_group) < 2:
            continue
        ap_pairs = list(combinations(current_group['combined_text'], 2))
        np.random.shuffle(ap_pairs)
        ap_pairs = ap_pairs[:samples_per_class]
        for anchor, positive in ap_pairs:
            negative_label = next(label_cycle)
            while negative_label == label:
                negative_label = next(label_cycle)
            if not other_groups[other_groups['label'] == negative_label].empty:
                negative_example = other_groups[other_groups['label'] == negative_label]['combined_text'].sample(1).values[0]
                triplets.append((anchor, positive, negative_example))

    triplet_df = pd.DataFrame(triplets, columns=['Anchor', 'Positive', 'Negative'])
    return triplet_df

def prepare_data():
    """
    Prepares the data by generating or loading the triplet dataset.

    Returns:
        train_examples, val_examples, test_examples: Examples for training, validation, and testing.
    """
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    triplet_data_path = os.path.join(data_dir, 'triplet_data.csv')
    
    if os.path.exists(triplet_data_path):
        print(f"Loading existing triplet data from {triplet_data_path}")
        triplet_df = pd.read_csv(triplet_data_path)
    else:
        print(f"Triplet data not found. Generating new triplet data.")
        processed_data_path = os.path.join(data_dir, 'processed_data.csv')
        if not os.path.exists(processed_data_path):
            raise FileNotFoundError(f"File not found: {processed_data_path}")
        df = pd.read_csv(processed_data_path)
        triplet_df = generate_triplet_data(df)
        triplet_df.to_csv(triplet_data_path, index=False)
        print(f"Triplet data has been saved to {triplet_data_path}")

    train_df, test_df = train_test_split(triplet_df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)  # 10% of train for validation

    def df_to_examples(df):
        return [InputExample(texts=[row['Anchor'], row['Positive'], row['Negative']]) for index, row in df.iterrows()]

    train_examples = df_to_examples(train_df)
    val_examples = df_to_examples(val_df)
    test_examples = df_to_examples(test_df)
    
    return train_examples, val_examples, test_examples

def finetune_roberta(train_examples, val_examples):
    """
    Fine-tunes a pre-trained RoBERTa model on triplet data and saves the embeddings.

    Args:
        train_examples: Training examples
        val_examples: Validation examples

    Returns:
        None
    """
    model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')

    # Define the triplet loss and data loaders
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    val_dataloader = DataLoader(val_examples, shuffle=False, batch_size=16)
    train_loss = losses.TripletLoss(model=model)

    # Initialize Weights and Biases
    os.environ["WANDB_DISABLED"] = "false"
    wandb.init(project="fine-tune-embeddings")

    # Define training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir="output/training_distilroberta_v1_triplet",
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if your GPU can't handle FP16
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        report_to="wandb",
        run_name="distilroberta-v1-triplet-run",
    )

    # Define an evaluator
    val_evaluator = evaluation.TripletEvaluator.from_input_examples(val_examples, name='val')

    # Fit the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=val_evaluator,
        epochs=args.num_train_epochs,
        evaluation_steps=args.eval_steps,
        warmup_steps=int(args.warmup_ratio * len(train_dataloader)),
        output_path=args.output_dir,
        save_best_model=True,
    )

    # Save the model to the specified directory
    model_save_path = os.path.join(os.path.dirname(__file__), '../models')
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Save the embeddings
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    texts = [example.texts[0] for example in train_examples] + [example.texts[1] for example in train_examples] + [example.texts[2] for example in train_examples]
    embeddings = model.encode(texts, convert_to_numpy=True)
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df['label'] = df['label']
    embeddings_df.to_csv(os.path.join(data_dir, 'embeddings_with_labels.csv'), index=False)
    print("Embeddings have been saved to '../data/embeddings_with_labels.csv'.")

if __name__ == "__main__":
    train_examples, val_examples, test_examples = prepare_data()
    finetune_roberta(train_examples, val_examples)
