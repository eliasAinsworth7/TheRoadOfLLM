# LLM nasıl kullanılır

Bilgi çağında, Uzun Kısa Hafıza (LLM) projeleri, bilgiyi derinlemesine anlamlandıran ve hızlı bir şekilde erişilebilir kılan güçlü araçlardır. Bu projeler, yapay zeka ve doğal dil işleme teknolojilerini kullanarak geniş konu yelpazelerinde veri toplar, analiz eder ve sonuçları sunar.

Bu repository de LLM projeleri oluşturmada kullanılan modellerden Huggin Face den GPT-3 hakkında bilgi vereceğim.

Hazır mısınız? O zaman, bilgi denizinde keyifli bir yolculuğa çıkalım ve LLM projelerinin büyüleyici dünyasını keşfedin!


# Hunggin face üzerinden GPT-3

Hugging Face üzerinden GPT-3 modelini kullanmak için aşağıdaki adımları izleyebilirsiniz:

1.Hugging Face Hesabı Oluşturun: Eğer henüz bir Hugging Face hesabınız yoksa, Hugging Face web sitesine giderek bir hesap oluşturun.
2.API Token Alın: Hugging Face hesabınıza giriş yaptıktan sonra, API token’ınızı almak için kullanıcı profilinize gidin. Bu token, Hugging Face API’sini kullanmanızı sağlayacak.
3.Gerekli Kütüphaneleri Yükleyin: Python ortamınızda transformers ve torch gibi gerekli kütüphaneleri yükleyin. Bu kütüphaneler, GPT-3 modelini kullanmanız için gereklidir.
```
pip install transformers torch
```

4.Modeli Yükleyin: transformers kütüphanesi aracılığıyla GPT-3 modelini yükleyin. Örneğin, aşağıdaki Python kodunu kullanabilirsiniz:
```
Python

from transformers import GPT3LMHeadModel, GPT3Tokenizer

model_name = 'gpt3' # Model adını buraya yazın

tokenizer = GPT3Tokenizer.from_pretrained(model_name)
model = GPT3LMHeadModel.from_pretrained(model_name)
```
5.Metin Üretimi Yapın: Modelinizi kullanarak metin üretimi yapabilirsiniz. Örneğin, bir metin parçası vererek modelinizden devamını üretmesini isteyebilirsiniz:
```
Python

input_text = "Bu bir örnek cümledir ve modelimiz"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```
6.Fine-Tuning Yapın: Eğer modelinizi belirli bir görev için özelleştirmek istiyorsanız, fine-tuning yapabilirsiniz. Bu, modelinizi kendi veri setinizle eğiterek gerçekleştirilir.
Bu adımlar, Hugging Face üzerinden GPT-3 modelini kullanmanıza yardımcı olacaktır. Daha fazla bilgi ve örnekler için Hugging Face’in resmi dökümantasyonuna ve model sayfasına göz atabilirsiniz.

Ve bir örnek... The Author is Neri Van Otten
```
import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments

# Define the task
task_name = "news_classification"
num_labels = 3

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load the pre-trained GPT-3 model
model = GPT2ForSequenceClassification.from_pretrained('EleutherAI/gpt-neo-1.3')

# Add a classification head on top of the model
model.resize_token_embeddings(len(tokenizer))
model.classifier = torch.nn.Linear(model.config.hidden_size, num_labels)

# Prepare the data
train_texts = ["News article 1", "News article 2", ...]
train_labels = [0, 1, ...]

val_texts = ["News article 101", "News article 102", ...]
val_labels = [0, 2, ...]

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encodings, train_labels)
val_dataset = NewsDataset(val_encodings, val_labels)

# Fine-tune the model
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# Deploy the model
model.save_pretrained('news_classification_model')
tokenizer.save_pretrained('news_classification_tokenizer')
```

