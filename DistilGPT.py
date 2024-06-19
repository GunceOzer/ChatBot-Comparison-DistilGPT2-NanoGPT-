import nltk
import pandas as pd
import torch
from datasets import Dataset
from nltk.translate import meteor_score
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, AdamW
from transformers import get_scheduler
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import numpy as np
from torch.utils.data import DataLoader
import json
from tqdm.auto import tqdm

# Load the processed dataset
file_path = '../../Downloads/cleaned_processed_conversation_pairs.csv'
df = pd.read_csv(file_path)

# Ensure the dataset has the necessary columns
if 'input' not in df.columns or 'target' not in df.columns:
    raise KeyError("The dataset must contain 'input' and 'target' columns.")

# Limit the dataset size for quick testing
datas = df.sample(n=20000, random_state=42)

# Concatenate input and target columns with a space separator
datas['input_target'] = datas['input'].fillna('') + ' [SEP] ' + datas['target'].fillna('')

# Reset index to avoid unexpected column in the dataset
datas = datas.reset_index(drop=True)

# Convert the DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(datas[['input_target']])

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

# Set padding token
tokenizer.pad_token = tokenizer.eos_token

# Determine max_length based on your analysis
max_length = 64  # Update this based on your distribution analysis

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['input_target'], padding="max_length", truncation=True, max_length=max_length)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["input_target"])

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Split the dataset into train and test
split_datasets = tokenized_datasets.train_test_split(test_size=0.1)
train_dataset = split_datasets["train"]
test_dataset = split_datasets["test"]

# Load the model
model = AutoModelForCausalLM.from_pretrained('distilgpt2')

# Move model to the appropriate device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Print the device being used
print(f"Using device: {device}")

# Training parameters
learning_rate = 5e-5
weight_decay = 0.01
train_batch_size = 100
eval_batch_size = 8
num_train_epochs = 10

# Prepare data loaders
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, collate_fn=data_collator)
eval_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, collate_fn=data_collator)

# Prepare optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
num_training_steps = num_train_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Function to compute metrics
def compute_metrics(pred_str, label_str):
    rouge = Rouge()
    rouge_output = rouge.get_scores(pred_str, label_str, avg=True)

    smoothing_function = SmoothingFunction().method4
    bleu_scores = []
    for ref, hyp in zip(label_str, pred_str):
        ref = ref.split()
        hyp = hyp.split()
        bleu_scores.append(sentence_bleu([ref], hyp, smoothing_function=smoothing_function))

    return {
        "rouge1": rouge_output["rouge-1"]["f"],
        "rouge2": rouge_output["rouge-2"]["f"],
        "rougel": rouge_output["rouge-l"]["f"],
        "bleu": np.mean(bleu_scores),
    }

# Training loop
model.train()
progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    eval_preds = []
    eval_labels = []
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            eval_preds.extend(predictions.cpu().numpy().tolist())
            eval_labels.extend(batch["labels"].cpu().numpy().tolist())
    
    # Check and clean eval_labels
    valid_eval_labels = []
    for label in eval_labels:
        valid_label = [token if 0 <= token < tokenizer.vocab_size else tokenizer.pad_token_id for token in label]
        valid_eval_labels.append(valid_label)

    # Decode predictions and labels
    eval_preds_str = tokenizer.batch_decode(eval_preds, skip_special_tokens=True)
    eval_labels_str = tokenizer.batch_decode(valid_eval_labels, skip_special_tokens=True)
    
    # Compute metrics
    metrics = compute_metrics(eval_preds_str, eval_labels_str)
    
    # Save metrics to file
    with open("../../Downloads/distilgpt2_results.json", "a") as f:
        json.dump({f"epoch_{epoch}": metrics}, f, indent=4)
    
    model.train()

# Save the trained model
model.save_pretrained("./results")
tokenizer.save_pretrained("./results")

# Prediction function
def generate_response(prompt, model, tokenizer, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load the saved model
model = AutoModelForCausalLM.from_pretrained('./results').to(device)

# Use specific samples from the dataframe for testing
test_samples = df.iloc[101:106]

for idx, row in test_samples.iterrows():
    prompt = row['input']
    reference = row['target']
    response = generate_response(prompt, model, tokenizer)
    print(f"Input: {prompt}")
    print(f"Response: {response}")
    print(f"Reference: {reference}")
    print("----")
