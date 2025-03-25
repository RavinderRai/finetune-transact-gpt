# Fine-tuning GPT 4 on Secure Data

Here we'll demo a simple implementation of a pipeline that takes in sensitive data and stores it securely, before ultimately fine-tune a gpt model on it. And well, it'll just be a kaggle data in this case, but the point is to introduce one method to handle data that may be sensitive yet required for a fine-tuning job. The process is as follows:

 - Ingest data
 - Generate synthetic data
 - Encrypt the data
 - Decrypt and fine-tune the data
 - Test inference with the new model

The dataset we'll use to demonstrate this is on financial transactions data, which you can get here: https://www.kaggle.com/datasets/ismetsemedov/transactions. Now you'll quickly notice it is not natural language data fit for a fine-tuning job with an LLM, hence why we have a synthetic data generation step in our pipeling - we'll use the tabular data to simulate natural language financial transaction data. Moreover, you'll also see a lot of features that we won't actually need as this is data meant for fraud detection, but we are juts adapting it for our use case.

Finally, you can find all relevant code files here: https://github.com/RavinderRai/finetune-transact-gpt. Make sure to pip install all requirements before testing it!

## Data Ingestion

So the first step is straightforward, as we just want to download the data. That being said, it becomes complicated in this case as the data is almost 3 GB with over 7 million samples, so you can't simply run basic pandas functions to explore it, but you can still load it by specifying the chunksize argument that pandas offers. The following code will give you the data.


```python
dataset_slug = "ismetsemedov/transactions"
file_name = "synthetic_fraud_data.csv"
chunk_size = 100_000

dataset_dir = kagglehub.dataset_download(dataset_slug)
file_path = os.path.join(dataset_dir, file_name)

chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)
df = pd.concat(chunk_iter, ignore_index=True)
```

Regardless, we are just demoing this type of problem so we can randomly sample a smaller subset. In fact, the target variable for our problem - merchant category - has only 8 categories, so we can randomly select 200 samples which will give us around 25 data points for each.


```python
output_path = "sampled_data.csv"

df_sample = df.sample(n=200, random_state=42)
df_sample.to_csv(output_path, index=False)
```

Now we have our tabular data, so we need to convert it to natural language text.

## Synthetic Dara Generation

As we all know, LLMs do not work well with tabular data, so we need to convert this. We can achieve this simply with different transaction templates and select them randomly for data variety. To do this we will want to get the relevant features and craft natural text with it, like so:


```python
merchant = row["merchant"]
amount = row["amount"]
currency = row["currency"]
city = row["city"]
country = row["country"]
timestamp = row["timestamp"]
card_type = row["card_type"]

templates = [
    f"{merchant} charged {amount:.2f} {currency} on your {card_type} card in {city}, {country} on {date_str}.",
    f"Transaction at {merchant} for {amount:.2f} {currency} in {city}, {country} ({card_type}) on {date_str}.",
    f"You spent {amount:.2f} {currency} at {merchant}, {city}, {country} using your {card_type} card on {date_str}.",
    f"Purchase from {merchant}: {amount:.2f} {currency}, paid with {card_type} in {city}, {country} on {date_str}.",
    f"{merchant} transaction of {amount:.2f} {currency} in {country} with {card_type} on {date_str}."
]
```

It should be noted that for more variety you could prompt an LLM to generate this data for you, but that might be overkill for this demo so we won't cover that here.

Then you can iterate on rows from a pandas dataframe and randomly select one of the templates to be the new data. Moreover, you can then store it in the proper format that openai fine-tuning jobs would want. This might look something like this:


```python
with open("finetune_data.jsonl", "w") as f_out:
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
```

Where generate description would be a function to perform the above action to get the natural langauge data. Then once we have it in this format, we can encrypt it.

## Data Encryption

So why encrypt it now. It's so that you are not storing raw sensitive data. This is "data-at-rest" protection, where we encrypt the whole jsonl file that has our training data before ultimately storing it in a cloud-like tool. Then, decryption will be performed right before uploading it to OpenAI (don't worry, you can delete it later) before fine-tuning. That way we only store the encrypted data.

Encryption can be done with the cryptography python library, where we not only encrypt the data but also generate an encryption key. Then you can use it to encrypt your data like so:


```python
from cryptography.fernet import Fernet

key = Fernet.generate_key()
fernet = Fernet(key)

encrypted_data = fernet.encrypt(data)
```

## Fine-Tuning with OpenAI

Now we can finally fine-tune our model on our dataset. You can find the documentation for the entire breakdown on how to do this here: https://platform.openai.com/docs/guides/fine-tuning?lang=python#preparing-your-dataset. Regarding LLM applications there isn't typically as much feature engineering you need to do as compared with traditional machine learning applications, so all you need to do is look at some examples on their link for how to prepare your data and match them. We already have our data ready though, so you can also look again at how we prepared our data, but for clarity's sake here is one sample row: 


```python
{
    "messages": 
    [
        {
            "role": "user", 
            "content": "Transaction: Red Lobster transaction of 544.43 BRL in Brazil with Basic Debit on October 02, 2024."
        }, 
        {
            "role": "assistant", 
            "content": "Restaurant"
        }
    ]
}
```

So our targets are just one word as this is a multi-categorical task. Next we first need to remember to decrypt our data. This is simple enough though, just get your key and run the decrypt function.


```python
# You'll might have relevant files saved locally if you're testing this, so make sure to open them
key_path = "encryption.key"
encrypted_path = "finetune_data_encrypted.bin"

with open(key_path, "rb") as f:
    key = f.read()

with open(encrypted_path, "rb") as f:
    encrypted_data = f.read()

fernet = Fernet(key)

decrypted_data = fernet.decrypt(encrypted_data)
```

Then we can start the fine-tuning job. Following openai's documentation, you can send a fine-tuning job to run on their servers with one of their models like this:


```python
# Make sure to set the proper variable names and file paths
jsonl_path = "finetune_data.jsonl"
model = "gpt-4o-mini-2024-07-18"
n_epochs=2

client = OpenAI()

training_file = client.files.create(
    file=open(jsonl_path, "rb"),
    purpose="fine-tune"
)

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
```

And just like that, your fine tuning job will be running. In this example with only 200 samples, you can expect it to only take a few minutes, but if you have a larger dataset be prepared for the training job to take a while. To see the progress though, simply go to this link: https://platform.openai.com/finetune/. From here, you can see the status of your fine-tuning job and also the id. Once it's done, we can test it out.

## Inference Testing

Finally we can do a quick inference test. To do this, run this code block:


```python
client = OpenAI()

response = client.chat.completions.create(
    model=model_id,
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0  # deterministic
)

prediction = response.choices[0].message.content.strip()
```

Where prompt is a sample data (you can just use one randomly from the training data), and the model_id is the name of the model you'll find here: https://platform.openai.com/finetune. For example, here are some inference tests done on the first few samples in the training set.

**Example 1**
- **Prompt:** Transaction: Red Lobster transaction of 544.43 BRL in Brazil with Basic Debit on October 02, 2024.
- **Predicted Category:** `Restaurant`
- **Actual Category:** `Restaurant`

**Example 2**
- **Prompt:** Transaction: Skillshare charged 1.46 CAD on your Basic Credit card in Unknown City, Canada on October 08, 2024.
- **Predicted Category:** `Education`
- **Actual Category:** `Education`

## Conclusion

That's it for this quick demonstration! We went from structured data to synthetic natural language, and then to data encryption before ultimately fine-tuning a gpt model. While the model was trained on just a small sample of data, you will see that it is still able to accurately categorize transactions based on natural language inputs — showing the power and practicality of LLM fine-tuning even on lightweight prototypes.

For a real production LLM system, this approach will naturally be extended with more complex components. Some things you might consider looking into include:

- **Human-in-the-loop feedback**: Letting users manually correct misclassifications and feeding those corrections back into the training data.
- **Live re-training or continual fine-tuning**: Periodically updating the model as new transaction types, merchants, or edge cases emerge.

And on the **data security** side, additional steps could might look like:

- **Field-level encryption**: Encrypting specific fields (e.g., card number, IP address) even within the training data, for stricter compliance.
- **Differential privacy**: Adding noise to training data or model outputs to limit the ability to reconstruct individual examples.

Thanks for reading!
