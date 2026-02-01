import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 1024  # change it to fit your memory constraints, e.g., 256, 128 if you run out of memory
EPOCHS = 25
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx] 
        context = row["context"]
        center = row["center"]
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)


# Simple Skip-gram Module
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(Word2Vec, self).__init__()
        self.centerEmbedding = nn.Embedding(vocab_size, embed_size)
        self.contextEmbedding = nn.Embedding(vocab_size, embed_size)
    def forward(self, center_labels, context_labels):
        center_vectors = self.centerEmbedding(center_labels)
        context_vectors = self.contextEmbedding(context_labels)
        if context_labels.dim() == 2:
            score = torch.bmm(center_vectors.unsqueeze(1), context_vectors.transpose(1, 2))
            return score.squeeze(1)
        else:
            score = torch.sum(center_vectors * context_vectors, dim = 1)
            return score
    def get_embeddings(self):
        return self.centerEmbedding.weight.data.cpu().numpy()


# Load processed data
with open("processed_data.pkl", "rb") as f:
    loaded_data = pickle.load(f)
sent_list = loaded_data['sent_list']
counter = loaded_data['counter']
word2idx = loaded_data['word2idx']
idx2word = loaded_data['idx2word']
skipgram_df = loaded_data['skipgram_df']

# Precompute negative sampling distribution below
word_counts = torch.zeros(len(word2idx))
for word, count in counter.items():
    if word in word2idx:
        word_counts[word2idx[word]] = count
pow_counts = torch.pow(word_counts, 0.75)
sampling_prob = pow_counts / pow_counts.sum()
# Device selection: CUDA > MPS > CPU
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(device)

# Dataset and DataLoader
dataset = SkipGramDataset(skipgram_df)
dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

# Model, Loss, Optimizer
vocab_size = len(word2idx)
model = Word2Vec(vocab_size, EMBEDDING_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
loss = nn.BCEWithLogitsLoss()

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0
    for center, context in dataloader:
        bs = center.shape[0]
        center = center.to(device)
        context = context.to(device)

        neg_context = torch.multinomial(sampling_prob, bs * NEGATIVE_SAMPLES, replacement=True)
        neg_context = neg_context.view(bs, NEGATIVE_SAMPLES).to(device)

        pos_score = model(center, context) 
        pos_label = torch.ones(bs).to(device)
        pos_loss = loss(pos_score, pos_label)

        neg_score = model(center, neg_context) 
        mask = (neg_context == context.unsqueeze(1)) 
        neg_score[mask] = -1e9
        neg_label = torch.zeros(bs, NEGATIVE_SAMPLES).to(device)
        neg_loss = loss(neg_score, neg_label)

        combined_loss = pos_loss + neg_loss

        optimizer.zero_grad()
        combined_loss.backward()
        optimizer.step()

        total_loss += combined_loss.item()
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

# Save embeddings and mappings
embeddings = model.get_embeddings()
with open('word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'word2idx': loaded_data['word2idx'], 'idx2word': loaded_data['idx2word']}, f)
print("Embeddings saved to word2vec_embeddings.pkl")
