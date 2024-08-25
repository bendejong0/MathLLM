
def encode(examples, tokenizer):
    inputs = tokenizer(examples['question'], padding="max_length", truncation=True, max_length=512)
    targets = tokenizer(examples['output'], padding="max_length", truncation=True, max_length=512)
    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': targets['input_ids']
    }

def preprocess():
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    dataset = load_dataset("csv", data_files="data/dataset.csv")

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")

    dataset = dataset.map(lambda examples: encode(examples, tokenizer), batched=True)

    print("Preprocessing success!")

    return dataset, tokenizer