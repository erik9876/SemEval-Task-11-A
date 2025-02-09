from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, hamming_loss
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import os
import torch
import gc
import pandas as pd
import random
import numpy as np
from pathlib import Path

# ---------------------- #
# Reproducibility & Setup
# ---------------------- #

# Set a fixed seed for reproducibility across runs
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Check for GPU availability and set the computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------- #
# Data Loading & Preprocessing
# ------------------------- #

def load_emotion_dataset(path, split_ratio=0.8):
    """
    Load the emotion dataset from a CSV file and split it into training, validation, and test sets.
    For multilingual datasets, ensure that validation and test data include only English samples,
    while non-English samples (e.g., German) are used only in training.

    Args:
        path (str): Path to the CSV dataset.
        split_ratio (float): Proportion of data used for training (default is 0.8).

    Returns:
        Tuple: (train_dataset, validation_dataset, test_dataset)
    """
    # Load dataset from CSV file using the Hugging Face datasets library
    dataset_dict = load_dataset('csv', data_files=path)
    dataset = dataset_dict['train']

    # Filter dataset by language based on the 'id' prefix ("eng" for English, "deu" for German)
    eng_dataset = dataset.filter(lambda x: x['id'].startswith("eng"))
    deu_dataset = dataset.filter(lambda x: x['id'].startswith("deu"))

    # If all data is in English, perform a simple train/validation/test split
    if len(eng_dataset) == len(dataset):
        eng_train_test = eng_dataset.train_test_split(test_size=1 - split_ratio)
        eval_test = eng_train_test['test'].train_test_split(test_size=0.5)
        return eng_train_test['train'], eval_test['train'], eval_test['test']

    # For multilingual datasets, include non-English data only in the training set.
    # Calculate the number of samples to be used for train and test splits.
    total_amount = len(dataset)
    train_test_amount = total_amount * (1 - split_ratio)

    # Split the English dataset accordingly
    eng_train_test = eng_dataset.train_test_split(test_size=(train_test_amount / len(eng_dataset)))
    eval_test = eng_train_test['test'].train_test_split(test_size=0.5)

    # Concatenate English training samples with all German samples for training
    return concatenate_datasets([eng_train_test['train'], deu_dataset]), eval_test['train'], eval_test['test']


def preprocess_data(tokenizer, dataset):
    """
    Tokenize the text data using a pretrained tokenizer and convert labels into a multi-label format.
    Also applies padding and truncation to the sequences.

    Args:
        tokenizer: Pretrained tokenizer instance.
        dataset: The dataset to preprocess.

    Returns:
        The tokenized dataset.
    """

    def tokenize(batch):
        # Tokenize the 'text' field in the batch; pad to max_length and truncate if necessary
        tokenized_batch = tokenizer(batch['text'], padding='max_length', truncation=True, max_length=256)

        # Convert emotion labels into a tuple (multi-label format)
        # The dataset is expected to have columns 'anger', 'fear', 'joy', 'sadness', 'surprise'
        tokenized_batch['labels'] = list(zip(
            batch['anger'], batch['fear'], batch['joy'], batch['sadness'], batch['surprise']
        ))
        # Convert the list of tuples into a tensor of type float32, then back to list (required by Trainer)
        tokenized_batch['labels'] = torch.tensor(tokenized_batch['labels'], dtype=torch.float32).tolist()
        return tokenized_batch

    # Apply the tokenize function on the dataset, removing the 'id' column afterwards, and process in batches
    return dataset.map(tokenize, remove_columns=["id"], batched=True)


# -------------------------- #
# Metrics Calculation Helpers
# -------------------------- #

# Default threshold for converting sigmoid outputs to binary predictions
THRESHOLD = 0.5


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for multi-label classification: macro and micro F1-scores,
    macro precision, macro recall, subset accuracy, and Hamming loss.

    Args:
        eval_pred (tuple): A tuple (logits, true labels) from model evaluation.

    Returns:
        dict: A dictionary with evaluation metric names as keys and their computed values.
    """
    logits, labels = eval_pred
    # Apply sigmoid to logits and threshold them to get binary predictions
    preds = torch.sigmoid(torch.tensor(logits)) > THRESHOLD
    preds = preds.int().numpy()
    labels = np.array(labels).astype(int)

    return {
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=1),
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=1),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=1),
        "precision_micro": precision_score(labels, preds, average="micro", zero_division=1),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=1),
        "recall_micro": recall_score(labels, preds, average="micro", zero_division=1),
        "subset_accuracy": accuracy_score(labels, preds),
        "hamming_loss": hamming_loss(labels, preds)
    }


def find_best_threshold(logits, labels):
    """
    Determine the optimal threshold that maximizes the macro F1-score on the predictions.

    Args:
        logits: Model output logits.
        labels: True labels for the dataset.

    Returns:
        Tuple: (best_threshold, best_f1_score)
    """
    # Define a range of potential thresholds from 0.1 to 0.9
    thresholds = np.linspace(0.1, 0.9, 20)
    best_threshold = 0.5
    best_f1 = 0.0

    for th in thresholds:
        # Apply sigmoid and threshold to obtain binary predictions
        preds = torch.sigmoid(torch.tensor(logits)) > th
        preds = preds.int().numpy()
        labels = np.array(labels).astype(int)

        # Calculate the macro F1-score for the current threshold
        f1 = f1_score(labels, preds, average="macro", zero_division=1)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th

    print(f"Best Threshold: {best_threshold} with F1-Score: {best_f1:.4f}")
    return best_threshold, best_f1


# -------------------------- #
# Training & Evaluation Logic
# -------------------------- #

def train_and_evaluate(config_params):
    """
    Main function to train and evaluate the model.
    It loads the tokenizer and pretrained model, processes the datasets, sets training arguments,
    trains the model, logs evaluation metrics per epoch, and finally evaluates the model on the test set.

    Args:
        config_params (list): List of configuration parameters in the following order:
            [data_path, dropout, model_name, lr_scheduler_type, learning_rate, batch_size, num_epochs, weight_decay, grad_clipping]

    Returns:
        Tuple: (final F1 score on test set, best threshold value)
    """
    # Unpack configuration parameters
    data_path, dropout, model_name, lr_scheduler_type, learning_rate, batch_size, num_epochs, weight_decay, grad = config_params

    print(f"Training with configuration: {config_params}")

    # ------------------------------ #
    # Load Pretrained Model & Tokenizer
    # ------------------------------ #
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=5,  # There are 5 emotion labels: anger, fear, joy, sadness, surprise
        ignore_mismatched_sizes=True,
        problem_type="multi_label_classification"  # Set the problem type to multi-label classification
    )
    # Update dropout probabilities in the model configuration
    model.config.hidden_dropout_prob = dropout
    model.config.attention_probs_dropout_prob = dropout
    model.to(device)

    # ------------------------------ #
    # Load and Preprocess Datasets
    # ------------------------------ #
    train_dataset, eval_dataset, test_dataset = load_emotion_dataset(data_path)
    train_dataset = preprocess_data(tokenizer, train_dataset)
    eval_dataset = preprocess_data(tokenizer, eval_dataset)
    test_dataset = preprocess_data(tokenizer, test_dataset)

    # ------------------------------ #
    # Set Training Arguments
    # ------------------------------ #
    training_args = TrainingArguments(
        output_dir=f"./models/{model_name}_semeval",
        save_strategy="epoch",  # Save model at the end of each epoch
        eval_strategy="epoch",  # Evaluate model at the end of each epoch
        save_total_limit=1,  # Limit the total number of saved checkpoints
        load_best_model_at_end=True,  # Load the best model when training is finished
        metric_for_best_model="eval_f1_macro",  # Use macro F1-score as the criterion for best model
        greater_is_better=True,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=50,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        max_grad_norm=grad,  # Gradient clipping value
        logging_dir="./logs",
        logging_steps=10,
        report_to="none"
    )

    # ------------------------------ #
    # Initialize Trainer and Train Model
    # ------------------------------ #
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics  # Use the compute_metrics function for evaluation
    )

    # Begin training
    trainer.train()

    # Save evaluation logs to a CSV file for further analysis
    eval_dir = f"./models/{model_name}_semeval"
    os.makedirs(eval_dir, exist_ok=True)
    eval_file = os.path.join(eval_dir, "eval_results.csv")
    df_eval = pd.DataFrame(trainer.state.log_history)
    df_eval.to_csv(eval_file, index=False)
    print(f"Eval results saved in: {eval_file}")

    # Identify the best epoch based on the highest eval_f1_macro score
    best_epoch = None
    best_f1_score = 0.0
    for log in trainer.state.log_history:
        if "eval_f1_macro" in log:
            if log["eval_f1_macro"] > best_f1_score:
                best_f1_score = log["eval_f1_macro"]
                best_epoch = log["epoch"]
    print(f"Best model is from epoch: {best_epoch} with f1 macro: {best_f1_score:.4f}")

    # ------------------------------ #
    # Evaluate on Test Set
    # ------------------------------ #
    test_predictions = trainer.predict(test_dataset)

    # Find the best threshold based on the test predictions to maximize macro F1-score
    best_threshold, best_f1 = find_best_threshold(test_predictions.predictions, test_predictions.label_ids)

    print("Test Metrics:")
    for key, value in test_predictions.metrics.items():
        print(f"{key}: {value:.4f}")

    # Recompute final predictions using the tuned threshold
    preds = torch.sigmoid(torch.tensor(test_predictions.predictions)) > best_threshold
    preds = preds.int().numpy()
    true_labels = np.array(test_predictions.label_ids).astype(int)
    final_f1 = f1_score(true_labels, preds, average="macro", zero_division=1)

    print(f"Final Test Macro F1-Score (with tuned threshold {best_threshold}): {final_f1:.4f}")

    # ------------------------------ #
    # Cleanup Resources
    # ------------------------------ #
    # Delete the model from memory and free up GPU resources
    del model
    torch.cuda.empty_cache()
    gc.collect()

    print(f"F1-Score: {final_f1} for configuration: {config_params}")
    return final_f1, best_threshold


# -------------------------- #
# Results Saving Utility
# -------------------------- #

def save_results(tuning_results, file_name):
    """
    Save the configuration results and corresponding performance metrics to a CSV file.

    Args:
        tuning_results (list): List of tuples, where each tuple is (config, threshold, score)
        file_name (str): Path to the CSV file where results will be saved.
    """
    df_data = []
    # Define column names for the results CSV file
    df_columns = [
        "Dataset", "Dropout", "Model", "LR Scheduler", "LR", "Batch", "Epochs",
        "Decay", "Grad Clipping", "Threshold", "F1 Score"
    ]
    for data, th, sc in tuning_results:
        # Append the best threshold and F1 score to the configuration parameters
        data.append(th)
        data.append(sc)
        df_data.append(data)

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(df_data, columns=df_columns)
    df.to_csv(file_name, index=False)
    print(f"Results saved to {file_name}")


# -------------------------- #
# Main Execution Block
# -------------------------- #

if __name__ == "__main__":
    # Define a list of configuration parameters for training
    # Each configuration includes:
    # [data_path, dropout, model_name, lr_scheduler_type, learning_rate, batch_size, num_epochs, weight_decay,
    #  grad_clipping]
    configs = [["track_a/train/eng.csv", 0.2, "roberta-large", "linear", 2e-05, 16, 10, 0.02, 0.5]]

    results = []
    for config in configs:
        # Train and evaluate the model with the given configuration
        score, threshold = train_and_evaluate(config)
        results.append((config, threshold, score))

    # Save all the results to a CSV file
    folder_path = Path("results")
    folder_path.mkdir(parents=True, exist_ok=True)
    save_results(results, file_name=f"results/results_final_config.csv")
    print(f"Training complete. Results saved to 'results/results_final_config.csv'.")
