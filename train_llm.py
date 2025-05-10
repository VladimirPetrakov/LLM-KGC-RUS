from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

model_name = "meta-llama/Meta-Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset("json", data_files="dataset/llm_finetune_dataset.jsonl", split="train")

def preprocess(example):
    return tokenizer(
        example["prompt"],
        text_target=example["completion"],
        truncation=True,
        max_length=512,
    )
dataset = dataset.map(preprocess, batched=True)

training_args = TrainingArguments(
    output_dir="./finetuned_model",
    per_device_train_batch_size=2,
    num_train_epochs=2,
    learning_rate=2e-5,
    fp16=True,
    save_steps=500,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")
