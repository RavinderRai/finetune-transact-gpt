import os
from cryptography.fernet import Fernet
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_key(key_path: str = "encryption.key") -> bytes:
    """
    Generates and saves an encryption key if not already present.
    """
    if not os.path.exists(key_path):
        key = Fernet.generate_key()
        with open(key_path, "wb") as f:
            f.write(key)
        logger.info(f"Encryption key saved to {key_path}")
    else:
        with open(key_path, "rb") as f:
            key = f.read()
        logger.info(f"Encryption key loaded from {key_path}")
    return key

def encrypt_file(input_path: str, output_path: str, key_path: str="encryption.key") -> None:
    """
    Encrypts the given file using Fernet symmetric encryption.
    """
    key = generate_key(key_path)
    fernet = Fernet(key)

    with open(input_path, "rb") as f:
        data = f.read()

    encrypted_data = fernet.encrypt(data)

    with open(output_path, "wb") as f:
        f.write(encrypted_data)

    logger.info(f"Encrypted data saved to {output_path}")

if __name__ == "__main__":
    input_file = "data/finetune_data.jsonl"
    encrypted_output = "data/finetune_data_encrypted.bin"
    
    if not os.path.exists(input_file):
        logger.error(f"Input file does not exist: {input_file}")
    else:
        encrypt_file(input_file, encrypted_output)