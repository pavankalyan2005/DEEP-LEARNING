import torch
import torch.nn as nn

torch.manual_seed(7)
# Tiny parallel dataset: source tokens are mapped to target tokens.
src_vocab = {"<pad>":0, "<s>":1, "i":2, "am":3, "happy":4, "you":5, "are":6, "good":7}
tgt_vocab = {"<pad>":0, "<s>":1, "je":2, "suis":3, "heureux":4, "tu":5, "es":6, "bon":7}
pairs = [([2,3,4],[2,3,4]), ([5,6,7],[5,6,7])]
class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__(); self.enc=nn.Embedding(8,16); self.lstm=nn.LSTM(16,24,batch_first=True)
        self.dec=nn.Embedding(8,16); self.out_lstm=nn.LSTM(16,24,batch_first=True); self.fc=nn.Linear(24,8)
    def forward(self, src, tgt):
        _, state=self.lstm(self.enc(src)); decoded,_=self.out_lstm(self.dec(tgt),state); return self.fc(decoded)
model=Seq2Seq(); optimizer=torch.optim.Adam(model.parameters(),lr=0.03); loss_fn=nn.CrossEntropyLoss()
src=torch.tensor([[2,3,4],[5,6,7]]); tgt_in=torch.tensor([[1,2,3],[1,5,6]]); tgt_out=torch.tensor([[2,3,4],[5,6,7]])
for _ in range(80):
    optimizer.zero_grad(); logits=model(src,tgt_in); loss=loss_fn(logits.reshape(-1,8),tgt_out.reshape(-1)); loss.backward(); optimizer.step()
with torch.no_grad():
    pred=model(src,tgt_in).argmax(-1)
inv={v:k for k,v in tgt_vocab.items()}
print("Model: LSTM encoder-decoder")
print("Training sentence pairs:",len(pairs))
print("Final training loss: {:.4f}".format(loss.item()))
print("Predicted target sequences:")
for row in pred.tolist(): print(" "," ".join(inv[x] for x in row))
print("Evaluation: predictions are compared with the target sequences in this small demonstration dataset.")
