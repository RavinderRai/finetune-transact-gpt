import os
from cryptography.fernet import Fernet
from openai import OpenAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KEY_PATH = "encryption.key"
ENCRYPTED_FILE = "data/finetune_data_encrypted.bin"
DECRYPTED_FILE = "data/finetune_data.jsonl"
MODEL_NAME = "gpt-4o-mini-2024-07-18"

def decrypt_file(encrypted_path: str, decrypted_path: str, key_path: str = KEY_PATH) -> None:
    """
    Decrypts the encrypted .bin file to a .jsonl file.
    """
    logger.info("Decrypting encrypted training data...")

    if not os.path.exists(key_path):
        raise FileNotFoundError(f"Encryption key not found at: {key_path}")
    if not os.path.exists(encrypted_path):
        raise FileNotFoundError(f"Encrypted data file not found at: {encrypted_path}")

    with open(key_path, "rb") as f:
        key = f.read()

    fernet = Fernet(key)

    with open(encrypted_path, "rb") as f:
        encrypted_data = f.read()

    decrypted_data = fernet.decrypt(encrypted_data)

    with open(decrypted_path, "wb") as f:
        f.write(decrypted_data)

    logger.info(f"Decrypted data written to {decrypted_path}")

def start_fine_tuning(jsonl_path: str, model: str = MODEL_NAME, n_epochs=2) -> None:
    """
    Uploads the training file and starts the fine-tuning job.
    """
    client = OpenAI()

    logger.info(f"Uploading training file: {jsonl_path}")
    training_file = client.files.create(
        file=open(jsonl_path, "rb"),
        purpose="fine-tune"
    )
    logger.info(f"File uploaded. File ID: {training_file.id}")

    logger.info(f"Starting fine-tuning job on model: {model}")
    job = client.fine_tuning.jobs.create(
        training_file=training_file.id,
        model=model,
        method={
            "type": "supervised",
            "supervised": {
                "hyperparameters": {"n_epochs": n_epochs}
            }
        }
    )
    logger.info(f"Fine-tuning job started. Job ID: {job.id}")
    logger.info("You can monitor job status with the OpenAI CLI or API.")


if __name__ == "__main__":
    decrypt_file(ENCRYPTED_FILE, DECRYPTED_FILE)
    start_fine_tuning(DECRYPTED_FILE)

    # # Optional: delete decrypted file for security
    # if os.path.exists(DECRYPTED_FILE):
    #     os.remove(DECRYPTED_FILE)
    #     logger.info(f"🧼 Decrypted file deleted after upload: {DECRYPTED_FILE}")
