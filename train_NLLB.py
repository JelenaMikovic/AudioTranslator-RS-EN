from torch.utils.data import Dataset
import sentencepiece
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
!pip install accelerate -U
from datasets import load_dataset
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
dataset = load_dataset("facebook/flores", "srp_Cyrl-eng_Latn")
train_dataset = dataset['dev'] 

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

tokenized_dataset = tokenizer(train_dataset['sentence_srp_Cyrl'], return_tensors='pt', truncation=True, padding=True)
tokenized_dataset2 = tokenizer(train_dataset['sentence_eng_Latn'], return_tensors='pt', truncation=True, padding=True)

class CustomSeq2SeqDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],
        }
train_dataset = CustomSeq2SeqDataset(
    input_ids=tokenized_dataset['input_ids'],
    attention_mask=tokenized_dataset['attention_mask'],
    labels=tokenized_dataset2['input_ids'].clone() 
)

training_args = Seq2SeqTrainingArguments(
    output_dir="./finetuned_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    predict_with_generate=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
trainer.save_model("/content/drive/MyDrive/")
