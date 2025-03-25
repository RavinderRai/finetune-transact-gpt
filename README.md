# Fine-Tune Transaction GPT

This project fine-tunes a GPT model using transaction data to help it better understand and respond to financial queries. Here’s what it does:

- **Data Ingestion**: Downloads and processes transaction datasets from KaggleHub, letting you sample the data for use.
- **Synthetic Data Generation**: Creates synthetic transaction data in a JSONL format that’s ready for fine-tuning.
- **Data Encryption**: Secures sensitive data files with encryption using the `cryptography` library, ensuring safety before training.
- **Fine-Tuning**: Fine-tunes a pre-trained GPT model with the prepared data, using the `fine_tuning.py` script to decrypt and start the process via OpenAI's API.
- **Inference Testing**: After fine-tuning, you can test the model with sample prompts using the `inference_test.py` module to see how well it generates relevant responses.
