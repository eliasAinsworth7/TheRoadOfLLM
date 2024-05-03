# LLM nasıl kullanılır

Büyük Dil Modelleri (LLM), genellikle büyük miktarda metin ve veri üzerinde eğitilmiş yapay zeka modelleridir ve çeşitli doğal dil işleme (NLP) görevlerinde kullanılır. LLM’leri kullanmak için genel adımlar şunlardır:

1.Hedefi Belirleyin: LLM’nin hangi görevde kullanılacağını belirleyin. Bu, metin üretimi, çeviri, özetleme veya başka bir NLP görevi olabilir1.
2.Ön Eğitim: LLM’yi eğitmek için büyük ve çeşitli bir veri kümesi gereklidir. Verilerin toplanması ve temizlenmesi, tüketim için standart hale getirilmesi gerekir1.
3.Tokenizasyon: LLM’nin kelimeleri veya alt kelimeleri anlayabilmesi için veri kümesindeki metni daha küçük birimlere ayırın1.
4.Altyapı Seçimi: LLM, eğitimin üstesinden gelmek için güçlü bir bilgisayar veya bulut tabanlı sunucu gibi hesaplama kaynaklarına ihtiyaç duyar1.
5.Eğitim: Eğitim süreci için yığın boyutu veya öğrenme oranı gibi parametreleri ayarlayın1.
6.İnce Ayar (Fine-Tuning): Eğitim yinelemeli bir süreçtir. Modele veri sunar, çıktısını değerlendirir ve ardından sonuçları iyileştirmek ve modele ince ayar yapmak için parametreleri ayarlar

# Fine-Tuning Nedir
Fine-tuning, yapay zeka ve makine öğrenimi alanında kullanılan bir terimdir ve “ince ayar” anlamına gelir. Bu işlem, genel bir modeli alıp onu belirli bir görev veya veri seti için özelleştirmek anlamına gelir1. Örneğin, bir dil modelini belirli bir konu üzerine daha iyi metinler üretecek şekilde eğitmek için fine-tuning yapabilirsiniz. Bu süreç, modelin önceden eğitilmiş ağırlıklarını, yeni veri setine uyum sağlayacak şekilde güncellemeyi içerir ve böylece modelin belirli bir göreve yönelik performansını artırır.

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

# Ve bir örnek... The Author is Neri Van Otten

İşte bir metin sınıflandırma görevi için GPT-3’ü nasıl fine-tune edebileceğinize dair bir örnek:

1.Görevi Tanımlayın: Farklı kategorilere, örneğin politika, spor ve eğlence gibi kategorilere göre haber makalelerini sınıflandırmak istiyoruz.

2.Verileri Hazırlayın: İlgili kategorilerle etiketlenmiş haber makaleleri veri setine ihtiyacımız var. Mevcut veri setlerini, örneğin Reuters Corpus’u kullanabilir veya farklı kaynaklardan haber makaleleri kazıyarak ve manuel olarak etiketleyerek kendi veri setimizi oluşturabiliriz.

3.Modeli Fine-Tune Edin: Metin sınıflandırması için GPT-3’ü fine-tune etmek için Hugging Face transformers kütüphanesini kullanabiliriz. Önceden eğitilmiş GPT-3 modelini yükleyecek ve modelin üstüne bir sınıflandırma başlığı ekleyeceğiz. Modeli, geri yayılım ve gradyan iniş tekniklerini kullanarak etiketlenmiş haber makaleleri üzerinde eğiteceğiz.

4.Performansı Değerlendirin: Model eğitildikten sonra, haber makalelerinin bir doğrulama seti üzerinde performansını değerlendireceğiz. Modelin performansını değerlendirmek için doğruluk, kesinlik, hatırlama ve F1 puanı metriklerini kullanabiliriz.

5.Modeli Ayarlayın: Performans değerlendirmesine dayanarak, öğrenme hızı veya parti boyutu gibi modelin hiperparametrelerini ayarlayabilir ve istenen performansı elde edene kadar modeli yeniden eğitebiliriz.

6.Modeli Dağıtın: Son olarak, haber makalelerini gerçek zamanlı olarak farklı kategorilere sınıflandırmak üzere üretimde modeli dağıtabiliriz.
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

