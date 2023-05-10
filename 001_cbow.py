import torch
import torch.nn as nn
import torch.optim as optim
from utils import process_w2v_data  # 这里指的是我的 [repo](https://github.com/MorvanZhou/NLP-Tutorials/) 中的 utils.py
from visual import show_w2v_word_embedding  # 这里指的是我的 [repo](https://github.com/MorvanZhou/NLP-Tutorials/) 中的 visual.py

corpus = [
    # 数字
    "5 2 4 8 6 2 3 6 4",
    "4 8 5 6 9 5 5 6",
    "1 1 5 2 3 3 8",
    "3 6 9 6 8 7 4 6 3",
    "8 9 9 6 1 4 3 4",
    "1 0 2 0 2 1 3 3 3 3 3",
    "9 3 3 0 1 4 7 8",
    "9 9 8 5 6 7 1 2 3 0 1 0",

    # 字母，期望 9 接近字母
    "a t g q e h 9 u f",
    "e q y u o i p s",
    "q o 9 p l k j o k k o p",
    "h g y i u t t a e q",
    "i k d q r e 9 e a d",
    "o p d g 9 s a f g a",
    "i u y g h k l a s w",
    "o l u y a o g f s",
    "o p i u y g d a s j d l",
    "u k i l o 9 l j s",
    "y g i s h k j l f r f",
    "i o h n 9 9 d 9 f a 9",
]

class CBOW(nn.Module):
    def __init__(self, v_dim, emb_dim):
        super(CBOW, self).__init__()
        self.v_dim = v_dim
        self.embeddings = nn.Embedding(v_dim, emb_dim)

        # 噪声对比估计
        self.nce_w = nn.Parameter(torch.randn(v_dim, emb_dim) * 0.1)
        self.nce_b = nn.Parameter(torch.zeros(v_dim))

        self.opt = optim.Adam(self.parameters(), 0.01)

    def forward(self, x):
        # x.shape = [n, skip_window*2]
        o = self.embeddings(x)          # [n, skip_window*2, emb_dim]
        o = torch.mean(o, dim=1)        # [n, emb_dim]
        return o

    def loss(self, x, y):
        embedded = self.forward(x)
        return torch.mean(
            torch.nn.functional.nce_loss(
                weight=self.nce_w, bias=self.nce_b, target=y.unsqueeze(1),
                input=embedded, num_negative_samples=5, num_classes=self.v_dim))

    def step(self, x, y):
        self.opt.zero_grad()
        loss = self.loss(x, y)
        loss.backward()
        self.opt.step()
        return loss.item()
    
    def train(model, data):
        for t in range(2500):
            bx, by = data.sample(8)
            bx = torch.LongTensor(bx)
            by = torch.LongTensor(by)
            loss = model.step(bx, by)
            if t % 200 == 0:
                print("step: {} | loss: {}".format(t, loss))


if __name__ == "__main__":
    d = process_w2v_data(corpus, skip_window=2, method="cbow")
    m = CBOW(d.num_word, 2)
    train(m, d)

    # 绘制
    show_w2v_word_embedding(m, d, "./visual/results/cbow.png")
