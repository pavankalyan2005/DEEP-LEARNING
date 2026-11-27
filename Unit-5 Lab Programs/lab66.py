import torch
import torch.nn as nn
torch.manual_seed(4)
# The target is the first value in a 25-step sequence (a long-range dependency).
X=torch.randn(300,40,1); y=X[:,0,:]
class Model(nn.Module):
    def __init__(self, kind):
        super().__init__(); self.rnn=nn.RNN(1,16,batch_first=True) if kind=='RNN' else nn.LSTM(1,16,batch_first=True); self.fc=nn.Linear(16,1)
    def forward(self,x):
        z=self.rnn(x)[0][:,-1,:]; return self.fc(z)
for kind in ['RNN','LSTM']:
    torch.manual_seed(4); model=Model(kind); opt=torch.optim.Adam(model.parameters(),lr=0.005); loss_fn=nn.MSELoss()
    for _ in range(400):
        opt.zero_grad(); loss=loss_fn(model(X),y); loss.backward(); opt.step()
    with torch.no_grad(): final=loss_fn(model(X),y).item()
    print(f"{kind} final training MSE: {final:.4f}")
print("Comparison: lower MSE indicates better retention of the early sequence value.")
