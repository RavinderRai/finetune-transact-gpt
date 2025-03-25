import os
from cryptography.fernet import Fernet
from openai import OpenAI
import logging

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)