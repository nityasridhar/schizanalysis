from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('csv', data_files={'train': '/Users/nityasridhar/Documents/schizophrenia_symptoms.csv'})

# Define label mapping
label_mapping = {
    "Positive Symptoms": 0,
    "Negative Symptoms": 1,
    "Cognitive Impairment": 2
}

# Add label column
def map_labels(example):
    example['label'] = label_mapping[example['label']]
    return example

dataset = dataset['train'].map(map_labels)

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='no',  # Disable evaluation
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # Reduce this value
    per_device_eval_batch_size=4,   # Reduce this value
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine-tuned-bert')
tokenizer.save_pretrained('./fine-tuned-bert')
