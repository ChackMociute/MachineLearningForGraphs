import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm

NAME = 'full'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


for k, v in torch.load('data.pt').items():
    globals()[k] = v.to(device)
targets = targets.reshape(-1, 83)

bs = 512
epochs = 600
repetitions = 20

with open(f'data/{NAME}.csv', 'w') as f:
    f.write('rep,epoch,losses\n')


for rep in range(repetitions):
    net = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 83)
        ).to(device)
    criterion = nn.MSELoss()
    optim = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-3)
    
    net.train()
    for e in range(epochs):
        indices = np.random.permutation(np.arange(len(targets)))
        losses = []
        for i in range(0, len(targets), bs):
            idx = indices[i:i+bs]
            y = net(x[idx]) + old[idx]
            loss = criterion(y, targets[idx])
            loss.backward()
            optim.step()
            losses.append(loss.detach().cpu())
        with open(f'data/{NAME}.csv', 'a') as f:
            f.write(''.join([f'{rep},{e},{l}\n' for l in losses]))
    torch.save(net, f"data/models/{NAME}{rep}.pt")