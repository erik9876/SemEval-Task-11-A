import shap
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import transformers
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    # Read the CSV file containing the dataset
    df = pd.read_csv("track_a/train/eng.csv")

    # Define the model path and tokenizer model name
    # Make sure to update the model_path to the correct checkpoint if needed
    model_path = "models/roberta-large_1.0/checkpoint-695"  # change to actual model path
    tokenizer_model = "roberta-large"

    # Set the computation device
    # Use Apple's MPS (Metal Performance Shaders) if available on Mac, otherwise use CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load the pretrained model and tokenizer from the specified path
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    # Move the model to the chosen device
    model.to(device)

    # Build a transformers pipeline for text classification using the loaded model and tokenizer
    pipe = transformers.pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        top_k=None,
    )

    # Define the names of the emotion classes
    class_names = ["anger", "fear", "joy", "sadness", "surprise"]

    # Create a SHAP explainer for the pipeline
    explainer = shap.Explainer(pipe, output_names=class_names)

    # Sample 20 random text samples from the 'text' column of the dataframe
    data = df["text"].sample(n=20, random_state=42).to_list()

    # Compute SHAP values for the sampled data using the explainer
    shap_values = explainer(data)

    # Generate a SHAP text plot for the computed SHAP values
    shap_explanation_prob = shap.plots.text(shap_values, display=False)

    folder_path = Path("shap")
    folder_path.mkdir(parents=True, exist_ok=True)

    # Save the probability-based SHAP explanation as an HTML file
    with open("shap/explanation_prob.html", "w") as file:
        file.write(shap_explanation_prob)

    # Create a second SHAP explainer that rescales model outputs to logits
    logit_explainer = shap.Explainer(
        shap.models.TransformersPipeline(pipe, rescale_to_logits=True),
        output_names=class_names
    )

    # Compute SHAP values based on logits for the same sampled data
    logit_shap_values = logit_explainer(data)

    # Generate a SHAP text plot for the logit-based SHAP values
    shap_explanation_logit = shap.plots.text(logit_shap_values, display=False)

    # Save the logit-based SHAP explanation as an HTML file
    with open("shap/explanation_logit.html", "w") as file:
        file.write(shap_explanation_logit)
