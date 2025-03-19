# train.py
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet

# 1. Data Preparation
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
context_tags = set()

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    
    context_required = intent.get('context_required', [])
    context_set = intent.get('context_set', [])
    context_tags.update(context_required + context_set)
    
    for pattern in intent['patterns']:
        tokens = tokenize(pattern)
        all_words.extend(tokens)
        xy.append((tokens, tag, context_required))

# Preprocess vocabulary
ignore_chars = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_chars]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
context_tags = sorted(context_tags)

# 2. Feature Engineering
X = []
y = []
mlb = MultiLabelBinarizer()
mlb.fit([context_tags])

for tokens, tag, contexts in xy:
    bow = bag_of_words(tokens, all_words)
    ctx_features = mlb.transform([contexts]).flatten()
    X.append(np.concatenate((bow, ctx_features)))
    y.append(tags.index(tag))

# 3. Dataset Splitting
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.15,
    random_state=42,
    stratify=y
)

# 4. Class Balancing
y_train_np = np.array(y_train)
class_counts = np.bincount(y_train_np)
class_weights = 1. / (class_counts + 1e-6)
samples_weights = class_weights[y_train_np]

sampler = WeightedRandomSampler(
    weights=samples_weights,
    num_samples=len(X_train),
    replacement=True
)

# 5. Dataset Preparation
X_train_np = np.array(X_train, dtype=np.float32)
y_train_np = np.array(y_train, dtype=np.int64)
X_val_np = np.array(X_val, dtype=np.float32)
y_val_np = np.array(y_val, dtype=np.int64)

train_dataset = TensorDataset(
    torch.from_numpy(X_train_np),
    torch.from_numpy(y_train_np)
)

val_dataset = TensorDataset(
    torch.from_numpy(X_val_np),
    torch.from_numpy(y_val_np)
)

# 6. Training Configuration
input_size = X_train_np.shape[1]
hidden_size = 64
output_size = len(tags)
batch_size = 16
max_epochs = 2000
patience = 50
min_delta = 0.0005

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=2 if torch.cuda.is_available() else 0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=len(val_dataset),
    shuffle=False
)

# 7. Model Initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(
    input_size=input_size,
    hidden_size=hidden_size,
    num_classes=output_size,
    context_size=len(context_tags),
    dropout_prob=0.2
).to(device)

# 8. Training Setup
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.2,
    patience=10,
    min_lr=1e-5
)

# 9. Training Loop
best_val_acc = 0.0
epochs_no_improve = 0

for epoch in range(max_epochs):
    # Training Phase
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device).float()
        labels = labels.to(device).long()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
    
    # Validation Phase
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        inputs, labels = next(iter(val_loader))
        inputs = inputs.to(device).float()
        labels = labels.to(device).long()
        
        outputs = model(inputs)
        val_loss = criterion(outputs, labels).item()
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
    
    # Metrics
    train_loss /= len(train_loader.dataset)
    val_acc = correct / len(val_loader.dataset)
    
    # Learning Rate Schedule
    scheduler.step(val_loss)
    
    # Early Stopping
    if val_acc > best_val_acc + min_delta:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_no_improve += 1
    
    # Progress Monitoring
    if (epoch + 1) % 5 == 0:
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1:4d} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Val Acc: {val_acc:.2%} | '
              f'LR: {lr:.2e}')
    
    if epochs_no_improve >= patience:
        print(f'\nEarly stopping at epoch {epoch+1}')
        break

# 10. Save Final Model
model.load_state_dict(torch.load('best_model.pth'))
torch.save({
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags,
    "context_tags": context_tags,
    "bow_size": len(all_words),
    "context_size": len(context_tags)
}, 'data.pth')

print(f'\nTraining complete. Best validation accuracy: {best_val_acc:.2%}')