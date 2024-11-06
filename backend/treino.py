import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Configuração do dispositivo para usar GPU se disponível
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Carregar o dataset fictício
df = pd.read_csv('incidents_dataset.csv')
df.dropna(subset=['descricao', 'categoria'], inplace=True)  # Remover linhas com valores nulos

# 2. Dividir os dados para treino e teste
X_train, X_test, y_train, y_test = train_test_split(df['descricao'], df['categoria'], test_size=0.2, random_state=42)

# Codificação das labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Função para carregar o modelo e o tokenizer
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))
    model.to(device)
    return model, tokenizer

# Função para preparar o dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item
    def __len__(self):
        return len(self.labels)

# Função para treinar o modelo
def train_model(model, tokenizer, X_train, y_train_encoded):
    train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=128)
    train_dataset = Dataset(train_encodings, torch.tensor(y_train_encoded))

    # Tokenize o conjunto de dados de teste (para avaliação durante o treinamento)
    test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=128)
    test_dataset = Dataset(test_encodings, torch.tensor(y_test_encoded))

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        evaluation_strategy="epoch",  # Avaliação a cada época
        save_strategy="epoch",
        logging_dir='./logs'
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset  # Adiciona o conjunto de dados de avaliação
    )
    
    trainer.train()
    return model


# Modelos a serem usados no ensemble
model_names = ['bert-base-uncased', 'albert-base-v2', 'roberta-base']
models = []
tokenizers = []

# Treinamento dos modelos BERT, ALBERT e RoBERTa
for model_name in model_names:
    print(f"Treinando o modelo: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name)
    trained_model = train_model(model, tokenizer, X_train, y_train_encoded)
    models.append(trained_model)
    tokenizers.append(tokenizer)

# Função para fazer previsões com ensemble
def predict_ensemble(models, tokenizers, X_test):
    predictions = []
    for i, model in enumerate(models):
        tokenizer = tokenizers[i]
        test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=128, return_tensors="pt")
        test_encodings = {k: v.to(device) for k, v in test_encodings.items()}
        with torch.no_grad():
            outputs = model(**test_encodings)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            predictions.append(preds)

    # Votação majoritária
    predictions = np.array(predictions)
    final_predictions = [np.bincount(predictions[:, i]).argmax() for i in range(predictions.shape[1])]
    return final_predictions

# 3. Fazer previsões com o ensemble e avaliar o modelo
ensemble_preds = predict_ensemble(models, tokenizers, X_test)
ensemble_preds_decoded = label_encoder.inverse_transform(ensemble_preds)

print(f"Accuracy: {accuracy_score(y_test, ensemble_preds_decoded)}")
print(classification_report(y_test, ensemble_preds_decoded))

# 4. Salvar cada modelo do ensemble
for i, model_name in enumerate(model_names):
    models[i].save_pretrained(f'{model_name}_ensemble')
    tokenizers[i].save_pretrained(f'{model_name}_ensemble')
    print(f"Modelo {model_name} salvo em '{model_name}_ensemble'")
