import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd


data = pd.read_csv("small_movies/ratings.csv")

np.random.seed(816)
mask = np.random.rand(len(data)) < 0.8

train = data[mask].copy()
val = data[~mask].copy()


#data encoding
def proc_col(col, train_col=None):
    if train_col is not None:
        uniq = train_col.unique()
    else:
        uniq = col.unique()
    
    name2idx = {o:i for i, o in enumerate(uniq)}

    return name2idx, np.array([name2idx.get(x, -1) for x in col]), len(uniq)

def encode_data(df, train=None):
    df = df.copy()
    for col_name in ["userId", "movieId"]:
        train_col = None
        if train is not None:
            train_col = train[col_name]
        
        _, col, _ = proc_col(df[col_name], train_col)
        df[col_name] = col
        df = df[df[col_name] >= 0]

    return df

df_train = encode_data(train)
df_val = encode_data(val)

embed = nn.Embedding(10, 3)

a = torch.LongTensor([[1,2,0,4,5,1]])

embed(a)

## matrix fatorization without dnn
class MatrixFatorization(nn.Module):
    def __init__(self, num_users, num_items, emb_size = 100):
        super(MatrixFatorization, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)

        self.user_emb.weight.data.uniform_(0, 0.05)
        self.item_emb.weight.data.uniform_(0, 0.05)

    def forward(self, u, v):
        u = self.user_emb(u)
        v = self.item_emb(v)

        return (u * v).sum(1)


## 어떻게 돌아가는지 살펴보기
users = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 6]
movies = [0, 1, 1, 2, 0, 1, 0, 3, 0, 3, 3, 1, 3]
ratings = [4, 5, 5, 3, 4, 4, 5, 2, 1, 4, 5, 1, 3]
columns = ['userId', 'movieId', 'rating']

test_df = pd.DataFrame(np.asarray([users, movies, ratings]).T, columns=columns)

num_users = len(test_df.userId.unique())
num_items = len(test_df.movieId.unique())
emb_size = len(test_df.columns)

users = torch.LongTensor(test_df.userId.values)
items = torch.LongTensor(test_df.movieId.values)
user_emb = nn.Embedding(num_users, emb_size)
item_emb = nn.Embedding(num_items, emb_size)

U = user_emb(users)
V = item_emb(items)

(U*V).sum(1)

## 본격적인 학습
num_users = len(data.userId.unique())
num_items = len(data.movieId.unique())

model = MatrixFatorization(num_users=num_users, num_items=num_items)



def validation_loss(model, unsqueeze=False):
    model.eval()
    users = torch.LongTensor(df_val.userId.values)
    items = torch.LongTensor(df_val.movieId.values)
    ratings = torch.FloatTensor(df_val.rating.values)
    if unsqueeze:
        ratings = ratings.unsqueeze(1)
    
    y_hat = model(users, items)
    loss = F.mse_loss(y_hat, ratings)
    return loss

def train_mf(model, epoch=30, lr=0.05, wd=0.0, unsqueeze=False):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    users = torch.LongTensor(df_train.userId.values)
    items = torch.LongTensor(df_train.movieId.values)
    ratings = torch.FloatTensor(df_train.rating.values)
    if unsqueeze:
        ratings = ratings.unsqueeze(1)
    for i in range(epoch):
        y_hat = model(users, items)
        loss = F.mse_loss(y_hat, ratings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        val_loss = validation_loss(model, unsqueeze)
        print(f"Epoch {i} | Loss: {loss.item():.3f} | Validation loss {val_loss.item():.3f}")

class NNCollabFiltering(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100, n_hidden=64):
        super(NNCollabFiltering, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.lin1 = nn.Linear(2 * emb_size , n_hidden)
        self.lin2 = nn.Linear(n_hidden, 1)
        self.drop = nn.Dropout(0.1)

        self.lin1.weight.data.uniform_(0, 0.05)
        self.lin2.weight.data.uniform_(0, 0.05)
        self.user_emb.weight.data.uniform_(0, 0.05)
        self.item_emb.weight.data.uniform_(0, 0.05)

    def forward(self, u, v):
        if not isinstance(u, torch.Tensor):
            u = torch.LongTensor(u)
        if not isinstance(v, torch.Tensor):
            v = torch.LongTensor(v)
        U = self.user_emb(u)
        V = self.item_emb(v)
        x = torch.cat([U, V], dim=1)
        x = self.drop(x)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

model = NNCollabFiltering(num_users, num_items)

train_mf(model, wd=1e-6, unsqueeze=True)

def get_top_ratings(user_num, num_movies, model, top=100):
    model.eval()
    user = torch.LongTensor([user_num] * num_movies)
    movie = torch.LongTensor(range(num_movies))

    ratings = model(user, movie).detach().numpy()
    rank = np.flip(np.argsort(ratings.squeeze()))
    
    return rank[:top]