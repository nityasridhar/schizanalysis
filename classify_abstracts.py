import os
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
import PyPDF2

def extract_text_from_pdf(pdf_path):
    print(f"Reading PDF: {pdf_path}")
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def extract_abstracts(text):
    abstracts = []
    abstract_start = text.lower().find('abstract')
    while abstract_start != -1:
        abstract_end = text.find('\n', abstract_start + 9)
        abstract = text[abstract_start:abstract_end].strip()
        abstracts.append(abstract)
        abstract_start = text.lower().find('abstract', abstract_end)
    print(f"Extracted {len(abstracts)} abstracts")  # Print the number of abstracts found
    return abstracts

def classify_abstract(abstract, classifier):
    predictions = classifier(abstract)
    return predictions

def count_categories_in_pdf(pdf_file, classifier, label_mapping):
    category_counts = {category: 0 for category in label_mapping.values()}
    pdf_path = os.path.expanduser(pdf_file)
    text = extract_text_from_pdf(pdf_path)
    abstracts = extract_abstracts(text)
    for abstract in abstracts:
        predictions = classify_abstract(abstract, classifier)
        print(f"Predictions: {predictions}")  # Print predictions
        for prediction in predictions:
            label = prediction['label']
            category = label_mapping.get(label)
            if category:
                category_counts[category] += 1
                print(f"Matched Category: {category}, Prediction: {label}")
    return category_counts

# Define the label mapping
label_mapping = {
    'LABEL_0': 'Positive Symptoms',
    'LABEL_1': 'Negative Symptoms',
    'LABEL_2': 'Cognitive Impairment'
}

# Load fine-tuned model and tokenizer
model_path = './fine-tuned-bert'
classifier = pipeline('text-classification', model=model_path, tokenizer=model_path)

# Specify the PDF file
pdf_file = '~/Downloads/2024_AnnCongress_AbstractBook-3-1268.pdf'

# Get the categorized abstracts
category_counts = count_categories_in_pdf(pdf_file, classifier, label_mapping)

# Display the results
print("Category Counts:")
for category, count in category_counts.items():
    print(f"{category}: {count} abstracts")
