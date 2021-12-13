from flask import Flask, jsonify, request

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical

app = Flask(__name__, static_url_path="/static")
device = "cpu"
model = None
optimizer = None

_GAMES = []


class GameDataset(Dataset):
    def __init__(self, split):
        self._data = []
        i = 0
        for g in _GAMES:
            winner = g[-1]["winner"]
            for d in g[:-1]:
                if winner == 1:
                    state = [
                        d["state"]["y2"],
                        d["state"]["y1"],
                        1.0 - d["state"]["x"],
                        d["state"]["y"],
                        -d["state"]["dx"],
                        d["state"]["dy"],
                    ]
                    if (split == "train" and i % 10 != 0) or (split == "test" and i % 10 == 0):
                        # if d["p1_action"] != 0:
                        self._data.append([state, d["p1_action"] + 1])
                if winner == 2:
                    state = [
                        d["state"]["y1"],
                        d["state"]["y2"],
                        d["state"]["x"],
                        d["state"]["y"],
                        d["state"]["dx"],
                        d["state"]["dy"],
                    ]
                    if (split == "train" and i % 10 != 0) or (split == "test" and i % 10 == 0):
                        # if d["p2_action"] != 0:
                        self._data.append([state, d["p2_action"] + 1])
                i += 1
        print(f"Loaded {len(self._data)} {split} items")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return torch.tensor(self._data[idx][0]), torch.tensor(self._data[idx][1])


@app.route("/move", methods=["POST"])
def move():
    global model
    d = request.json
    temperature = d["temperature"]
    state = [
        d["state"]["y1"],
        d["state"]["y2"],
        d["state"]["x"],
        d["state"]["y"],
        d["state"]["dx"],
        d["state"]["dy"],
    ]
    x = torch.tensor(state)
    action = 0
    with torch.no_grad():
        pred = model(x)
        m = Categorical(logits=pred / temperature)
        action = m.sample().item() - 1

    return jsonify(
        {
            "action": action,
        }
    )


@app.route("/end_game", methods=["POST"])
def end_game():
    log = request.json
    _GAMES.append(log)
    return jsonify(
        {
            "recorded_games_count": len(_GAMES),
        }
    )


@app.route("/train", methods=["POST"])
def train():
    global model
    global optimizer

    req = request.json

    lr = req['learning_rate']
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr
        print(f"setting lr={lr} pg={i}")

    train_data = GameDataset(split="train")
    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    train_size = len(train_dataloader.dataset)
    train_num_batches = len(train_dataloader)
    train_loss, correct = 0, 0

    for batch, (X, y) in enumerate(train_dataloader):
        pred = model(X)
        # print(f"pred={pred}, y={y}")
        loss = loss_fn(pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= train_num_batches
    print(f"Train Error: \n Avg loss: {train_loss:>8f} \n")

    test_data = GameDataset(split="test")
    test_dataloader = DataLoader(test_data, batch_size=8, shuffle=True)

    test_size = len(test_dataloader.dataset)
    test_num_batches = len(test_dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in test_dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= test_num_batches
    correct /= test_size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return jsonify({
        "train_size": train_size,
        "train_batches": train_num_batches,
        "train_loss": train_loss,
        "test_loss": test_loss,
    })


@app.route("/reset", methods=["POST"])
def reset():
    reset_model()
    return jsonify({})


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


def reset_model():
    print("Resetting model")
    global model
    global optimizer
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


if __name__ == "__main__":
    reset_model()
    print(model)

    app.run(host="0.0.0.0", debug=True)
