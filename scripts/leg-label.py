#!/usr/bin/env python3
"""
Legislative Bill Classification Script
Tests 3 models x 3 prompt strategies = 9 combinations
"""

import os
import re
import pandas as pd
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_prompt_template(prompt_file: str) -> str:
    """Load a prompt template from the prompts directory"""
    prompt_path = Path("prompts") / prompt_file
    with open(prompt_path, 'r') as f:
        return f.read()

def classify_bill(description: str, prompt_template: str, model: str, is_reasoning: bool = False) -> tuple[int | None, dict]:
    """
    Classify a single bill using a specific model and prompt
    Returns: (predicted_category, usage_stats)
    """
    # Format the prompt with the bill description
    prompt = prompt_template.format(description=description)

    try:
        # Call the API with more tokens for reasoning prompts
        max_tokens = 200 if is_reasoning else 10

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=max_tokens
        )

        # Extract response text
        response_text = response.choices[0].message.content.strip()

        # Parse the category number from the response
        # For reasoning prompts, look for "ANSWER: X" pattern first
        if is_reasoning:
            match = re.search(r'ANSWER:\s*(\d+)', response_text, re.IGNORECASE)
            if match:
                prediction = int(match.group(1))
            else:
                # Fallback: look for any number at the end of the text
                match = re.search(r'\b([1-9]|1[0-9]|20)\b(?!.*\b([1-9]|1[0-9]|20)\b)', response_text)
                if match:
                    prediction = int(match.group(1))
                else:
                    print(f"Could not parse category from response: {response_text[:50]}")
                    prediction = None
        else:
            # For simple prompts, look for a number between 1 and 20
            match = re.search(r'\b([1-9]|1[0-9]|20)\b', response_text)
            if match:
                prediction = int(match.group(1))
            else:
                print(f"Could not parse category from response: {response_text}")
                prediction = None

        # Extract usage stats
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        return prediction, usage
    except Exception as e:
        print(f"Error classifying: {e}")
        return None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def run_experiment(data_file: str = "data/leg_lite.feather",
                   sample_size: int | None = None):
    """
    Run the full 3�3 experimental design

    Args:
        data_file: Path to the legislative bills dataset
        sample_size: If provided, only test on first N bills (for testing)
    """
    # Load data
    print("Loading data...")
    df = pd.read_feather(data_file)

    if sample_size:
        df = df.head(sample_size)
        print(f"Testing on {sample_size} bills")
    else:
        print(f"Processing {len(df)} bills")

    # Define experimental conditions
    models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]  # Small, optimized, and turbo models
    prompts = {
        "simple": load_prompt_template("cap-simple.md"),
        "detailed": load_prompt_template("cap-detailed.md"),
        "reasoning": load_prompt_template("cap-reasoning.md")
    }

    # Initialize results storage
    results = df[['id', 'description', 'policy', 'policy_label']].copy()

    # Track token usage
    usage_stats = {}

    # Run all 9 combinations
    for model in models:
        for prompt_name, prompt_template in prompts.items():
            col_name = f"{model.replace('gpt-', 'gpt').replace('-turbo-preview', '').replace('-', '_')}_{prompt_name}"
            print(f"\nProcessing: {model} + {prompt_name} prompt")

            predictions = []
            total_tokens = 0

            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{model}_{prompt_name}"):
                is_reasoning = (prompt_name == "reasoning")
                pred, usage = classify_bill(row['description'], prompt_template, model, is_reasoning)
                predictions.append(pred)
                total_tokens += usage['total_tokens']

            # Store results
            results[col_name] = predictions
            usage_stats[col_name] = total_tokens

            print(f"Total tokens used: {total_tokens:,}")

    # Calculate accuracy for each combination
    print("\n" + "="*60)
    print("ACCURACY RESULTS")
    print("="*60)

    for col in results.columns:
        if col not in ['id', 'description', 'policy', 'policy_label']:
            correct = (results[col] == results['policy']).sum()
            total = len(results)
            accuracy = correct / total * 100
            tokens = usage_stats[col]
            print(f"{col:50s}: {accuracy:5.2f}% ({correct}/{total}) | Tokens: {tokens:,}")

    # Export results
    output_file = "data/leg_predictions.feather"
    results.to_feather(output_file)
    print(f"\nResults saved to: {output_file}")

    # Also save usage stats
    usage_df = pd.DataFrame([usage_stats])
    usage_df.to_csv("data/token_usage.csv", index=False)
    print(f"Token usage saved to: data/token_usage.csv")

    return results, usage_stats

if __name__ == "__main__":
    # Run on full dataset
    print("Running on full dataset (500 bills)...")
    print("This will take 30-60 minutes and will consume API tokens.")
    print("Estimated cost: $5-20 depending on the models used.\n")
    run_experiment()
