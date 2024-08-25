from transformers import TrainingArguments
import evaluate
from preprocess import preprocess
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


data,tokenizer=preprocess()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = tokenizer.batch_decode(logits, skip_special_tokens=True)
    references = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return metric.compute(predictions=predictions, references=references)

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large", num_labels=5)

metric = evaluate.load("rouge")

training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir='./logs'
)


trainer = Trainer(
    model=AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large"),
    args=training_args,
    train_dataset=data['train'],
    eval_dataset=data.get('validation'),
    compute_metrics=compute_metrics
)

trainer.train()