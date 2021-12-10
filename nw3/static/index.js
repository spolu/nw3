const _BOARD_WIDTH = 800;
const _BOARD_HEIGHT = 600;

const _PLAYER_HEIGHT = 80;
const _PLAYER_STATE_HALFSPAN = (_PLAYER_HEIGHT / 2) / _BOARD_HEIGHT;

const _BALL_RADIUS = 10;

const _LOOP_SLEEP = 70;
const _ACTION_DELTA = 0.04;
const _BALL_DELTA = 0.02;
const _BALL_PLAYER_DELTA_X = 0.005;

var _RUNNING = null;
var _NEXT_GAME = 0;

var _GAME_STATE = {
  "y1": 0.5,
  "y2": 0.5,
  "x": 0.5,
  "y": 0.5,
  "dx": 0,
  "dy": 0,
};

var _GAME_LOG = [];

var _P1_ACTION = 0;

let sleep = (ms) => {
  return new Promise(resolve => setTimeout(resolve, ms));
};

let display = () => {
  // console.log(_GAME_STATE);
  $('#p1').css({
    top: (_GAME_STATE["y1"] * _BOARD_HEIGHT - _PLAYER_HEIGHT / 2) + 'px',
  });
  $('#p2').css({
    top: (_GAME_STATE["y2"] * _BOARD_HEIGHT - _PLAYER_HEIGHT / 2) + 'px',
  });
  $('#ball').css({
    left: (_GAME_STATE["x"] * _BOARD_WIDTH - _BALL_RADIUS / 2) + 'px',
    top: (_GAME_STATE["y"] * _BOARD_HEIGHT - _BALL_RADIUS / 2) + 'px',
  });
};

let clear = async() => {
  _GAME_STATE = {
    "y1": 0.5,
    "y2": 0.5,
    "x": 0.5,
    "y": 0.5,
    "dx": 0,
    "dy": 0,
  };
  _GAME_LOG = [];
};

let update_player = (player, action) => {
  const p = `y${player}`;
  if (action == 1) {
    _GAME_STATE[p] = _GAME_STATE[p] - _ACTION_DELTA;
  }
  if (action == -1) {
    _GAME_STATE[p] = _GAME_STATE[p] + _ACTION_DELTA;
  }
  _GAME_STATE[p] = Math.max(_GAME_STATE[p], 0);
  _GAME_STATE[p] = Math.min(_GAME_STATE[p], 1.0);
};

let end_game = (winner) => {
  console.log(`end_game: game=${_RUNNING} winner=${winner} states=${_GAME_LOG.length}`);

  _GAME_LOG.push({
    "game": _RUNNING,
    "type": "end_game",
    "winner": winner,
  });

  // console.log(_GAME_LOG);
  $.ajax({
    type: "POST",
    url: "/end_game",
    data : JSON.stringify(_GAME_LOG),
    contentType : 'application/json',
    success: (data) => {
      console.log(`uploaded_game: recorded_games_count=${data['recorded_games_count']}`);
    },
  });

  _RUNNING = null;
  clear();
};

let loop = async () => {
  let start_loop = (new Date()).getTime();
  // console.log(`loop start: start=${start_loop}`);

  // Get p1 command.
  const a1 = _P1_ACTION;
  _P1_ACTION = 0;
  // console.log(`loop p1_action: action=${a1}`);

  // Get p2 command (blocking).
  var a2 = 0;
  $.ajax({
    type: "POST",
    url: "/move",
    data : JSON.stringify(_GAME_STATE),
    contentType : 'application/json',
    success: (data) => {
      a2 = data['action'];
    },
    async: false,
  });

  // Log game state before updating.
  _GAME_LOG.push({
    "type": "loop",
    "game": _RUNNING,
    "state": {..._GAME_STATE},
    "p1_action": a1,
    "p2_action": a2,
  });

  // Apply physics.
  update_player(1, a1);
  update_player(2, a2);

  if (_GAME_STATE["y"] >= 1 || _GAME_STATE["y"] <= 0) {
    _GAME_STATE["dy"] = -_GAME_STATE["dy"];
  }
  if (_GAME_STATE["x"] <= 0) {
    if (Math.abs(_GAME_STATE["y1"] - _GAME_STATE["y"]) <= _PLAYER_STATE_HALFSPAN) {
      _GAME_STATE["dx"] = -_GAME_STATE["dx"] + _BALL_PLAYER_DELTA_X;
    } else {
      end_game(2);
    }
  }
  if (_GAME_STATE["x"] >= 1) {
    if (Math.abs(_GAME_STATE["y2"] - _GAME_STATE["y"]) <= _PLAYER_STATE_HALFSPAN) {
      _GAME_STATE["dx"] = -_GAME_STATE["dx"] - _BALL_PLAYER_DELTA_X;
    } else {
      end_game(1);
    }
  }
  _GAME_STATE["x"] = _GAME_STATE["x"] + _GAME_STATE["dx"];
  _GAME_STATE["y"] = _GAME_STATE["y"] + _GAME_STATE["dy"];

  // Update display.
  display();

  let duration = Math.max(0, _LOOP_SLEEP - ((new Date()).getTime() - start_loop));
  // console.log(`sleep: duration=${duration}`)
  await sleep(duration);

  let end_loop = (new Date()).getTime();
  // console.log(`loop end: end=${end_loop} duration=${end_loop-start_loop}`);
};

let start_game = async () => {
  clear();
  display();

  var angle = Math.random() * Math.PI / 4;
  if (Math.random() < 0.5) {
    angle = -angle;
  }
  if (Math.random() < 0.5) {
    angle += Math.PI;
  }
  _GAME_STATE["dx"] = _BALL_DELTA * Math.cos(angle);
  _GAME_STATE["dy"] = _BALL_DELTA * Math.sin(angle);

  const game = _NEXT_GAME;
  _NEXT_GAME = _NEXT_GAME + 1;

  console.log(`start_game: game=${game} angle=${angle}`);

  _RUNNING = game;
  while (_RUNNING == game) {
    await loop();
  }
};

let train = async () => {
  $.ajax({
    type: "POST",
    url: "/train",
    data : JSON.stringify({}),
    contentType : 'application/json',
    success: (data) => {
      console.log(data);
    },
    async: false,
  });
};

let reset = async () => {
  $.ajax({
    type: "POST",
    url: "/reset",
    data : JSON.stringify({}),
    contentType : 'application/json',
    success: (data) => {
      console.log(data);
    },
    async: false,
  });
}

(async () => {
  clear();
  display();
})();

document.addEventListener('keydown', (event) => {
  const code = event.keyCode;
  if (code == 38) {
    _P1_ACTION = 1
  }
  if (code == 40) {
    _P1_ACTION = -1
  }
  if (code == 83) { // s
    start_game();
  }
  if (code == 84) { // t
    train();
  }
  if (code == 82) { // r
    reset();
  }
});
