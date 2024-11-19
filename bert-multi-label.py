import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from utilities import convertStringListToString, read_json_from_folder

# Step 1: Read and preprocess your data
folder_path = 'ECHR_Dataset/EN_train'
folder_path2 = 'ECHR_Dataset/EN_test'

# Assuming a function that reads your JSON data into a DataFrame
train_data = read_json_from_folder(folder_path)
test_data = read_json_from_folder(folder_path2)

# Combine the data from both folders into one list
combined_data = train_data + test_data
df = pd.DataFrame(combined_data)
print("Completed framing data into df")

# Preprocess the TEXT and labels
df['TEXT'] = df['TEXT'].apply(convertStringListToString)
df['VIOLATED_ARTICLES'] = df['VIOLATED_ARTICLES'].apply(
    lambda x: x if isinstance(x, list) and len(x) > 0 else ['0']
)

# Assume the label column is `VIOLATED_ARTICLES`, where each row contains a list of labels
# Multi-label binarization (you need to transform labels into a binary format)
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['VIOLATED_ARTICLES'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['TEXT'], y, test_size=0.2, random_state=42)

# Step 2: Tokenize the texts using BERT tokenizer
print("Started tokenizer")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def tokenize_function(text):
    encoding = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    encoding['input_ids'] = encoding['input_ids'].long()
    encoding['attention_mask'] = encoding['attention_mask'].long()
    return encoding


# Tokenize the texts
train_tokens = X_train.apply(tokenize_function)
test_tokens = X_test.apply(tokenize_function)

# Prepare the inputs and labels for the model
train_inputs = {
    'input_ids': torch.cat([item['input_ids'] for item in train_tokens]).long(),
    'attention_mask': torch.cat([item['attention_mask'] for item in train_tokens]).long()
}
test_inputs = {
    'input_ids': torch.cat([item['input_ids'] for item in test_tokens]).long(),
    'attention_mask': torch.cat([item['attention_mask'] for item in test_tokens]).long()
}

train_labels = torch.tensor(y_train, dtype=torch.float32)  # Multi-label, hence float32
test_labels = torch.tensor(y_test, dtype=torch.float32)

# Step 3: Create DataLoader for training and testing
train_data = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
test_data = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], test_labels)

train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=8)

# Step 4: Initialize the BERT model for multi-label classification
print("Started Bert Classification")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=y_train.shape[1])

# Step 5: Move model to GPU if available
print("Setting device to gpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 6: Optimizer and Loss Function (BCEWithLogitsLoss for multi-label classification)
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = BCEWithLogitsLoss()  # For multi-label classification

# Mixed Precision Training (using GradScaler)
scaler = GradScaler()

# Step 7: Training Loop
epochs = 5
accumulation_steps = 2  # Gradient accumulation steps to simulate larger batch size

print("Started model training epochs")
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_dataloader):
        print(f"Processing batch {batch_idx + 1} of epoch {epoch + 1}")
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        # Forward pass with autocasting for mixed precision
        with autocast():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        # Backward pass with mixed precision
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()  # Update the scale for the next iteration
            optimizer.zero_grad()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss}")


# Step 8: Evaluation on training and test set

def get_predictions(dataloader):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Sigmoid activation to convert logits to probabilities
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()  # Threshold at 0.5 for multi-label

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return predictions, true_labels


# Get predictions for the training set
print("Calculating train predictions")
train_predictions, train_true_labels = get_predictions(train_dataloader)

# Get predictions for the test set
print("Calculating test predictions")
test_predictions, test_true_labels = get_predictions(test_dataloader)

# Step 9: Classification Reports

train_report = classification_report(train_true_labels, train_predictions, target_names=mlb.classes_)
test_report = classification_report(test_true_labels, test_predictions, target_names=mlb.classes_)

print("Classification Report for Train Data:")
print(train_report)

print("Classification Report for Test Data:")
print(test_report)

# Step 10: Save the model and tokenizer for future use
model.save_pretrained('./bert-multi-label-classification')
tokenizer.save_pretrained('./bert-multi-label-classification')

# Save optimizer state
torch.save(optimizer.state_dict(), './bert-multi-label-classification/optimizer.pth')


def plot_confusion_matrix(true_labels, predictions):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Violation (0)', 'Violation (1)'],
                yticklabels=['No Violation (0)', 'Violation (1)'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


# Plot confusion matrix for test predictions
plot_confusion_matrix(test_true_labels, test_predictions)

# Optionally, print confusion matrix values
cm = confusion_matrix(test_true_labels, test_predictions)
print("Confusion Matrix:")
print(cm)
