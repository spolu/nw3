from flask import Flask, jsonify, request

import torch
from torch import nn

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
                    self._data.append([state, d["p2_action"] + 1])

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx][0], self._data[idx][1]


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


@app.route("/move", methods=["POST"])
def move():
    x = torch.tensor(list(request.json.values()))
    action = 0
    with torch.no_grad():
        pred = model(x)
        action = pred.argmax(0).item() - 1

    return jsonify(
        {
            "action": action,
        }
    )


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == "__main__":
    print("Initializing network")
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print(model)

    app.run(host="0.0.0.0", debug=True)
