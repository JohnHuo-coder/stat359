import os
import gc
import random
import numpy as np
import pandas as pd
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
from gensim.models import KeyedVectors
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score
import gensim.downloader as api
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


LR = 0.002
EPOCHS = 30
WARMUP_EPOCHS = 5
BATCH_SIZE = 256
NUM_CLASSES = 3
TRAIN = False

def set_seed(seed=42):
    """
    Set seed for reproducibility
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

nltk_data_path = r'C:\Users\huoji\AppData\Roaming\nltk_data'
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

def tokenize_text(text):
    return word_tokenize(text.lower())

    
def load_local_gensim_model(model_name):
    """
    Load a gensim model from disk by name (without .model extension).
    If the model file does not exist, download and save it first.
    Example: model = load_local_gensim_model('word2vec-google-news-300')
    """
    path = f"{model_name}.model"
    if not os.path.exists(path):
        print(f"Model file {path} not found. Downloading '{model_name}'...")
        model = api.load(model_name)
        model.save(path)
        print(f"Model '{model_name}' downloaded and saved as '{path}'")
    print(f"Loading model from {path} ...")
    return KeyedVectors.load(path)

fasttext_model = load_local_gensim_model("fasttext-wiki-news-subwords-300")

def get_sentence_ft_embedding(sentence, max_len = 32):
    words = tokenize_text(sentence)[:max_len]
    result_embeddings = []
    for word in words:
        try:
            result_embeddings.append(fasttext_model[word])
        except:
            result_embeddings.append(np.zeros(300))
    while len(result_embeddings) < max_len:
        result_embeddings.append(np.zeros(300))

    return np.array(result_embeddings, dtype = np.float32)
    
def preprocess_for_mlp(batch):
    embeddings = [get_sentence_ft_embedding(s) for s in batch["sentence"]]
    return {"embeddings": torch.tensor(np.array(embeddings))}


dataset = load_dataset('financial_phrasebank', 'sentences_50agree', trust_remote_code=True)
split_dataset = dataset["train"].train_test_split(test_size=0.15, stratify_by_column="label", seed=42)
test_dataset = split_dataset["test"]
train_val_dataset = split_dataset["train"]
split_dataset_train_val = train_val_dataset.train_test_split(test_size=0.15, stratify_by_column="label", seed=42)
train_dataset = split_dataset_train_val["train"]
val_dataset = split_dataset_train_val["test"]

train_dataset = train_dataset.map(preprocess_for_mlp, batched=True)
test_dataset = test_dataset.map(preprocess_for_mlp, batched=True)
val_dataset = val_dataset.map(preprocess_for_mlp, batched=True)

train_dataset.set_format(type="torch", columns=["embeddings", "label"])
test_dataset.set_format(type="torch", columns=["embeddings", "label"])
val_dataset.set_format(type="torch", columns=["embeddings", "label"])


class LSTM_model(nn.Module):
    def __init__(self, emb_dim, hidden_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=1, 
                            batch_first=True, bidirectional=True, dropout=0.5)
        pooled_dim = 4 * hidden_dim
        
        self.ln = nn.LayerNorm(pooled_dim)
        self.fc = nn.Linear(pooled_dim, num_classes)
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        mask = (x.abs().sum(dim=-1) != 0).float().unsqueeze(-1) # [batch, 32, 1]

        sum_out = torch.sum(lstm_out * mask, dim=1)
        lengths = mask.sum(dim=1).clamp(min=1) 
        avg_pool = sum_out / lengths

        max_pool, _ = torch.max(lstm_out, dim=1)
        combined = torch.cat([avg_pool, max_pool], dim=1) 

        x = self.ln(combined)
        x = self.dropout(x)
        return self.fc(x)

def train_model(model, train_loader, device, criterion, optimizer):
    model.train()
    all_labels = []
    all_preds = []
    total_loss = 0
    cnt = 0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
    for step, batch in pbar:
        input = batch["embeddings"].to(device)
        label = batch["label"].to(device)
        output = model(input)
        loss = criterion(output, label)
        preds = torch.argmax(output, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(label.detach().cpu().numpy())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) 
        optimizer.step()
        total_loss += loss.item()
        cnt+=1
    avg_loss = total_loss/cnt
    macro_f1_score = f1_score(all_labels, all_preds, average="macro")
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    train_acc = np.sum(all_labels_np == all_preds_np) / len(all_labels)
    return avg_loss, macro_f1_score, train_acc

def evaluate(model, val_loader, device, criterion):
    model.eval()
    all_labels = []
    all_preds = []
    val_loss = 0
    cnt = 0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            inputs = batch["embeddings"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            val_loss += loss.item()
            cnt+=1
    avg_val_loss = val_loss/cnt
    macro_f1_score = f1_score(all_labels, all_preds, average="macro")
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    val_acc = np.sum(all_labels_np == all_preds_np) / len(all_labels)

    return avg_val_loss, macro_f1_score, val_acc

def run_test():
    all_labels = []
    all_preds = []

    test_loss = 0
    cnt = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM_model(300, 128, NUM_CLASSES).to(device)
    checkpoint_path = "outputs/lstm_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle = False)

    counts = torch.tensor([604, 2879, 1363], dtype=torch.float)
    weights = 1.0/counts
    weights = weights / weights.sum() * NUM_CLASSES
    weights = weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    with torch.no_grad():
        for batch in tqdm(test_loader):
            inputs = batch["embeddings"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            test_loss += loss.item()
            cnt+=1
    avg_test_loss = test_loss/cnt
    macro_f1_score = f1_score(all_labels, all_preds, average="macro")
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    test_acc = np.sum(all_labels_np == all_preds_np) / len(all_labels)
    print("\nTest F1 score:")
    print(macro_f1_score)

    print("\nTest Loss:")
    print(avg_test_loss)

    print("\nTest Accuracy:")
    print(test_acc)

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    plot_confusion_matrix(
        all_labels,
        all_preds,
        class_names=["Neg", "Neu", "Pos"],
        save_path="outputs/lstm_test_confusion_matrix_normalized.png",
        normalize=True
    )

    plot_confusion_matrix(
        all_labels,
        all_preds,
        class_names=["Neg", "Neu", "Pos"],
        save_path="outputs/lstm_test_confusion_matrix.png",
        normalize=False
    )

    plot_f1_loss_acc_bar(
        avg_test_loss,
        macro_f1_score,
        test_acc,
        save_path = "outputs/lstm_test_metric.png"
    )

def plot_f1_loss_acc_bar(
    loss,
    f1,
    acc,
    save_path):

    df = pd.DataFrame({
        "metric": ["loss", "macro f1", "accuracy"],
        "value": [loss, f1, acc]
    }
    )

    plt.figure(figsize=(6, 5))
    sns.barplot(
        x = "metric",
        y="value",
        data = df
    )

    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.title("Test Metric")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Test Metric saved to {save_path}")


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    save_path="confusion_matrix.png",
    normalize=False,
    cmap="Blues"
):

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Confusion matrix saved to {save_path}")


def plot_metrics(train_data, val_data, metric_name, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_data, label=f'Train {metric_name}')
    plt.plot(val_data, label=f'Val {metric_name}')
    plt.title(f'{metric_name} Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close() 

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    counts = torch.tensor([604, 2879, 1363], dtype=torch.float)
    weights = 1.0/counts
    weights = weights / weights.sum() * NUM_CLASSES
    weights = weights.to(device)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle = False)

    base_lr = LR

    model = LSTM_model(300, 128, NUM_CLASSES).to(device)
    optimizer = AdamW(model.parameters(), lr = base_lr, weight_decay=0.01)

    scheduler1 = LinearLR(optimizer, start_factor=0.1, total_iters = WARMUP_EPOCHS)
    scheduler2 = CosineAnnealingLR(optimizer, T_max = (EPOCHS-WARMUP_EPOCHS), eta_min=1e-3)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[WARMUP_EPOCHS])
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    all_train_losses = []
    all_train_f1 = []
    all_train_acc = []
    all_val_losses = []
    all_val_f1 = []
    all_val_acc = []

    best_f1 = 0
    best_epoch = 0
    for epoch in range(EPOCHS):
        avg_loss, train_f1_score, train_acc = train_model(model, train_loader, device, criterion, optimizer)
        all_train_losses.append(avg_loss)
        all_train_f1.append(train_f1_score)
        all_train_acc.append(train_acc)
        avg_val_loss, val_f1_score, val_acc = evaluate(model, val_loader, device, criterion)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} done. Current Learning Rate: {current_lr:.6f}")

        all_val_losses.append(avg_val_loss)
        all_val_f1.append(val_f1_score)
        all_val_acc.append(val_acc)

        print(f"Train Loss: {avg_loss:.4f}, Train F1: {train_f1_score:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1_score:.4f}, Val Accuracy: {val_acc:.4f}")
        
        if val_f1_score > best_f1:
            best_f1 = val_f1_score
            best_epoch = epoch + 1
            print(f"New best F1 {best_f1:.4f} at epoch {best_epoch}")

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_F1': val_f1_score,
                'train_F1': train_f1_score
            }, "outputs/lstm_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'val_F1': val_f1_score,
        'train_F1': train_f1_score
    }, "outputs/lstm_last_model.pth")
        
    plot_metrics(all_train_losses, all_val_losses, 'Loss', 'outputs/lstm_loss_curve.png')
    plot_metrics(all_train_f1, all_val_f1, 'Macro F1', 'outputs/lstm_f1_curve.png')
    plot_metrics(all_train_acc, all_val_acc, 'Accuracy', 'outputs/lstm_accuracy_curve.png')
    # Clear memory
    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()
    
    print("\n" + "="*60)
    print("Results:")
    print(f"Epoch {best_epoch}: {best_f1:.4f}")
    print("="*60)


if __name__ == "__main__":
    if TRAIN:
        print("\nStarting training...")
        run_training()
        print("\nTraining complete!")
    else:
        print("\nStarting Testing...")
        run_test()
        print("\nTesting complete!")
