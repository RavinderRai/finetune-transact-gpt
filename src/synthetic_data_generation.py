import pandas as pd
import os
import json
import random
from datetime import datetime
import logging

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_description(row: pd.Series) -> str:
    """
    Generates a natural language transaction description from a row.
    """
    merchant = row["merchant"]
    amount = row["amount"]
    currency = row["currency"]
    city = row["city"]
    country = row["country"]
    timestamp = row["timestamp"]
    card_type = row["card_type"]

    # Format date
    try:
        date_str = datetime.fromisoformat(str(timestamp)).strftime("%B %d, %Y")
    except Exception:
        date_str = "a recent date"
        logger.warning("Failed to parse timestamp, using default date.")

    # Random style variation
    templates = [
        f"{merchant} charged {amount:.2f} {currency} on your {card_type} card in {city}, {country} on {date_str}.",
        f"Transaction at {merchant} for {amount:.2f} {currency} in {city}, {country} ({card_type}) on {date_str}.",
        f"You spent {amount:.2f} {currency} at {merchant}, {city}, {country} using your {card_type} card on {date_str}.",
        f"Purchase from {merchant}: {amount:.2f} {currency}, paid with {card_type} in {city}, {country} on {date_str}.",
        f"{merchant} transaction of {amount:.2f} {currency} in {country} with {card_type} on {date_str}."
    ]

    return random.choice(templates)

def generate_finetune_jsonl(df: pd.DataFrame, output_path: str = "data/finetune_data.jsonl") -> None:
    """
    Generates OpenAI fine-tuning JSONL from DataFrame.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Generating fine-tuning JSONL to: {output_path}")

    with open(output_path, "w") as f_out:
        for _, row in df.iterrows():
            description = generate_description(row)
            category = row["merchant_category"]
            example = {
                "messages": [
                    {"role": "user", "content": f"Transaction: {description}"},
                    {"role": "assistant", "content": category}
                ]
            }
            f_out.write(json.dumps(example) + "\n")

    logger.info(f"Generated {len(df)} examples to: {output_path}")

# Example usage
if __name__ == "__main__":
    df_sampled = pd.read_csv("data/sampled_data.csv")
    logger.info("Starting to generate fine-tuning data.")
    generate_finetune_jsonl(df_sampled)
    logger.info("Finished generating fine-tuning data.")
