from flask import Flask, jsonify, request

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

app = Flask(__name__, static_url_path="/static")
device = "cpu"

_GAMES = []


class GameDataset(Dataset):
    def __init__(self, start, end):
        self._data = []
        for g in _GAMES[start:end]:
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
                    if d["p1_action"] != 0:
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
                    if d["p2_action"] != 0:
                        self._data.append([state, d["p2_action"] + 1])
        print(f"Loaded {len(self._data)} items")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return torch.tensor(self._data[idx][0]), torch.tensor(self._data[idx][1])


@app.route("/move", methods=["POST"])
def move():
    d = request.json
    state = [
        d["y1"],
        d["y2"],
        d["x"],
        d["y"],
        d["dx"],
        d["dy"],
    ]
    x = torch.tensor(state)
    action = 0
    with torch.no_grad():
        pred = model(x)
        action = pred.argmax(0).item() - 1

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

    training_data = GameDataset(start=0, end=len(_GAMES))
    dataloader = DataLoader(training_data, batch_size=8, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        print(f"pred={pred}, y={y}")
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

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


if __name__ == "__main__":
    print("Initializing network")
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    print(model)

    app.run(host="0.0.0.0", debug=True)
