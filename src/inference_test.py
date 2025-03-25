from openai import OpenAI
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONFIG ===
FINE_TUNED_MODEL = "ft:gpt-4o-mini-2024-07-18:personal::BEzwC0ea"  # 🔁 Replace with your actual model ID
JSONL_PATH = "data/finetune_data.jsonl"
NUM_SAMPLES = 3

def load_sample_prompts(path: str, n: int = 3) -> list:
    prompts = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            example = json.loads(line)
            for message in example["messages"]:
                if message["role"] == "user":
                    prompts.append(message["content"])
    return prompts

# === Run Inference ===
def run_inference(prompt: str, model_id: str = FINE_TUNED_MODEL) -> str:
    client = OpenAI()

    logger.info(f"Sending input to model: {prompt}")
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0  # deterministic
    )

    prediction = response.choices[0].message.content.strip()
    logger.info(f"Model response: {prediction}")
    return prediction

if __name__ == "__main__":
    prompts = load_sample_prompts(JSONL_PATH, NUM_SAMPLES)

    for i, prompt in enumerate(prompts, 1):
        logger.info(f"\n[{i}] Prompt: {prompt}")
        prediction = run_inference(prompt, FINE_TUNED_MODEL)
        print(f"Prediction: {prediction}\n")
