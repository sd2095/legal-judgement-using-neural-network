import pandas as pd
import torch
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from transformers import BertTokenizer, BertForSequenceClassification
from utilities import read_json_from_folder, convertStringListToString
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

folder_path = 'ECHR_Dataset/EN_train'
folder_path2 = 'ECHR_Dataset/EN_test'

# Read JSON files from both folders
print("Started reading json files for train data")
train_data = read_json_from_folder(folder_path)
print("Started reading json files for test data")
test_data = read_json_from_folder(folder_path2)

# Combine the data from both folders into one list
combined_data = train_data + test_data
df = pd.DataFrame(combined_data)
print("Completed framing data into df")

# Removing outliers
df['TEXT_LENGTH'] = df['TEXT'].apply(lambda x: len(' '.join(x)))
df = df[df['TEXT_LENGTH'] <= df['TEXT_LENGTH'].quantile(0.998)]
df['TEXT'] = df['TEXT'].apply(convertStringListToString)

print("Started tokenizer")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def tokenize_function(text):
    encoding = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    encoding['input_ids'] = encoding['input_ids'].long()  # Cast input_ids to torch.long
    encoding['attention_mask'] = encoding['attention_mask'].long()
    return encoding


tokens = df['TEXT'].apply(tokenize_function)

X_train, X_test, y_train, y_test = train_test_split(tokens, df.IMPORTANCE.astype(int).sub(1), test_size=0.2,
                                                    random_state=42)

train_inputs = {
    'input_ids': torch.cat([item['input_ids'] for item in X_train]).long(),
    'attention_mask': torch.cat([item['attention_mask'] for item in X_train]).long()
}

test_inputs = {
    'input_ids': torch.cat([item['input_ids'] for item in X_test]).long(),
    'attention_mask': torch.cat([item['attention_mask'] for item in X_test]).long()
}

train_labels = torch.tensor(y_train.values, dtype=torch.long)
test_labels = torch.tensor(y_test.values, dtype=torch.long)

print("Started Bert Classification")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

train_data = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
test_data = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], test_labels)

train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=8)

print("Setting device to gpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class_weights = compute_class_weight('balanced', classes=df.IMPORTANCE.astype(int).sub(1).unique(),
                                     y=df.IMPORTANCE.astype(int).sub(1).values)
class_weights = torch.tensor(class_weights, dtype=torch.float)

optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = CrossEntropyLoss(weight=class_weights)

epochs = 5
accumulation_steps = 2
scaler = GradScaler()

print("Started model training epochs")
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_dataloader):
        print(f"Processing {batch_idx} number batch of {epoch} epoch")
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        # Enable autocasting for the forward pass (use FP16 precision where possible)
        with autocast():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        # Backward pass with mixed precision
        scaler.scale(loss).backward()  # Scales the loss for stability

        # Gradient accumulation (if applicable)
        if (batch_idx + 1) % accumulation_steps == 0:
            # Unscale gradients before optimizer step
            scaler.step(optimizer)
            scaler.update()  # Update the scale for the next iteration
            optimizer.zero_grad()  # Reset gradients

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss}")

from sklearn.metrics import classification_report

model.eval()
train_predictions, train_true_labels = [], []

with torch.no_grad():
    for batch in train_dataloader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        preds = torch.argmax(logits, dim=1)
        train_predictions.extend(preds.cpu().numpy())
        train_true_labels.extend(labels.cpu().numpy())
        print("Calculating train predictions")

# Get predictions for the test set
test_predictions, test_true_labels = [], []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        preds = torch.argmax(logits, dim=1)
        test_predictions.extend(preds.cpu().numpy())
        test_true_labels.extend(labels.cpu().numpy())
        print("Calculating test predictions")

# Generate classification reports
train_report = classification_report(train_true_labels, train_predictions, target_names=[str(i) for i in range(4)])
test_report = classification_report(test_true_labels, test_predictions, target_names=[str(i) for i in range(4)])

# Print the reports
print("Classification Report for Train Data:")
print(train_report)

print("Classification Report for Test Data:")
print(test_report)

model.save_pretrained('./bert-multi-class-classification-case-imp')
tokenizer.save_pretrained('./bert-multi-class-classification-case-imp')

torch.save(optimizer.state_dict(), './bert-multi-class-classification-case-imp/optimizer.pth')


def plot_confusion_matrix(true_labels, predictions):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['1', '2', '3', '4'],
                yticklabels=['1', '2', '3', '4'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix of Case Importance')
    plt.show()


# Plot confusion matrix for test predictions
plot_confusion_matrix(test_true_labels, test_predictions)

# Optionally, print confusion matrix values
cm = confusion_matrix(test_true_labels, test_predictions)
print("Confusion Matrix:")
print(cm)
