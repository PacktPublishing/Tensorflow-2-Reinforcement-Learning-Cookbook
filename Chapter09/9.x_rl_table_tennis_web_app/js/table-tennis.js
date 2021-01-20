const WINNER_SCORE = 5;
const PADDLE_HEIGHT = 100;
const PADDLE_THICKNESS = 10;

var canvas,
  context,
  ballX = 50,
  ballSpeedX = 15,
  ballY = 10,
  ballSpeedY = 4,
  paddleRedY = 250,
  paddleBlueY = 250,
  player1Score = 0,
  player2Score = 0,
  done = false;

function calculateMousePos(event) {
  var rect = canvas.getBoundingClientRect(),
    root = document.documentElement;
  mouseX = event.clientX - rect.left - root.scrollLeft;
  mouseY = event.clientY - rect.top - root.scrollTop;
  return {
    x: mouseX,
    y: mouseY,
  };
}

function handleMouseClick(event) {
  if (done) {
    player1Score = 0;
    player2Score = 0;
    done = false;
  }
}

function handleMouseMove(event) {
  var mousePos = calculateMousePos(event);
  paddleRedY = mousePos.y - PADDLE_HEIGHT / 2;
}

function handleTouchStart(event) {
  handleMouseClick(event.touches[0]);
}

function handleTouchMove(event) {
  handleMouseMove(event.touches[0]);
  event.preventDefault();
}

function resizeTable() {
  var gameArea = document.getElementById("gameArea");
  var aspectRatio = 4 / 3;
  var newWidth = window.innerWidth;
  var newHeight = window.innerHeight;
  var newAspectRatio = newWidth / newHeight;

  if (newAspectRatio > aspectRatio) {
    newWidth = newHeight * aspectRatio;
    gameArea.style.height = newHeight + "px";
    gameArea.style.width = newWidth + "px";
  } else {
    newHeight = newWidth / aspectRatio;
    gameArea.style.width = newWidth + "px";
    gameArea.style.height = newHeight + "px";
  }

  gameArea.style.marginTop = -newHeight / 2 + "px";
  gameArea.style.marginLeft = -newWidth / 2 + "px";

  var gameCanvas = document.getElementById("ttPlayArea");
  gameCanvas.width = newWidth;
  gameCanvas.height = newHeight;
}

window.onload = function () {
  canvas = document.getElementById("ttPlayArea");
  context = canvas.getContext("2d");

  // Resize Table Canvas
  resizeTable();
  // Draw TT Board
  ttBoardCanvas = document.getElementById("ttBoard");
  ttBoardContext = ttBoardCanvas.getContext("2d");
  var ttBoard = new Image();
  ttBoard.src = "rsc/tt-board.svg";

  ttBoard.onload = function () {
    ttBoardContext.drawImage(
      ttBoard,
      0,
      0,
      ttBoardCanvas.width,
      ttBoardCanvas.height
    );
  };

  var framesPerSecond = 30;
  var baselineAgent = new BaselineAgent(PADDLE_HEIGHT);
  // Run env loop
  setInterval(function () {
    // Get Agent's action
    action = baselineAgent.act({ paddleBlueY: paddleBlueY, ballY: ballY });
    step(action);
    render();
  }, 1000 / framesPerSecond);

  // Mouse calbacks for user interaction
  canvas.addEventListener("mousedown", handleMouseClick, false);
  canvas.addEventListener("mousemove", handleMouseMove, false);

  // Touch callbacks for user interaction
  canvas.addEventListener("touchstart", handleTouchStart, false);
  canvas.addEventListener("touchmove", handleTouchMove, false);

  window.addEventListener("resize", resizeTable, false);
  window.addEventListener("orientationchange", resizeTable, false);
};

function ballReset() {
  if (player1Score >= WINNER_SCORE || player2Score >= WINNER_SCORE) {
    done = true;
  }

  ballSpeedX = -ballSpeedX;
  ballX = canvas.width / 2;
  ballY = canvas.height / 2;
}

class BaselineAgent {
  constructor(paddleHeight) {
    this.paddleHeight = paddleHeight;
    this.action = 0;
  }

  act(state) {
    var paddleBlueYCenter = state.paddleBlueY + this.paddleHeight / 2;
    if (paddleBlueYCenter < state.ballY + 0.3 * this.paddleHeight) {
      this.action = +6;
    } else {
      this.action = -6;
    }
    return this.action;
  }
}

function step(action) {
  if (done) {
    return;
  }

  // Apply Agent's action
  paddleBlueY += action;
  ballY += ballSpeedY;
  ballX += ballSpeedX;

  if (ballX > canvas.width) {
    if (ballY > paddleBlueY && ballY < paddleBlueY + PADDLE_HEIGHT) {
      ballSpeedX = -ballSpeedX;
      var deltaY = ballY - (paddleBlueY + PADDLE_HEIGHT / 2);
      ballSpeedY = deltaY * 0.35;
    } else {
      player1Score++;
      ballReset();
    }
  }

  if (ballX < 0) {
    if (ballY > paddleRedY && ballY < paddleRedY + PADDLE_HEIGHT) {
      ballSpeedX = -ballSpeedX;
      var deltaY = ballY - (paddleRedY + PADDLE_HEIGHT / 2);
      ballSpeedY = deltaY * 0.35;
    } else {
      player2Score++;
      ballReset();
    }
  }

  if (ballY > canvas.height) {
    ballSpeedY = -ballSpeedY;
  }

  if (ballY < 0) {
    ballSpeedY = -ballSpeedY;
  }
}

function drawNet() {
  for (var i = 0; i < canvas.height; i += 40) {
    colorRect(canvas.width / 2 - 1, i, 2, 20, "white");
  }
}

function render() {
  context.font = "20px Arial";
  context.textAlign = "center";
  // Draw the Table
  // colorRect(0, 0, canvas.width, canvas.height, "black");
  context.clearRect(0, 0, canvas.width, canvas.height); // TODO: clear only modified region
  if (done) {
    context.fillStyle = "white";

    if (player1Score >= WINNER_SCORE) {
      context.fillText(
        "Red Player Won!",
        canvas.width / 2,
        (canvas.height * 1) / 4
      );
    } else if (player2Score >= WINNER_SCORE) {
      context.fillText(
        "Blue Player Won!",
        canvas.width / 2,
        (canvas.height * 1) / 4
      );
    }
    context.fillText(
      "Click/Touch to continue",
      canvas.width / 2,
      canvas.height / 2
    );

    return;
  }

  drawNet();

  colorRect(0, paddleRedY, PADDLE_THICKNESS, PADDLE_HEIGHT, "red");
  colorRect(
    canvas.width - PADDLE_THICKNESS,
    paddleBlueY,
    PADDLE_THICKNESS,
    PADDLE_HEIGHT,
    "blue"
  );
  colorCircle(ballX, ballY, 10, "orange");
  context.fillText(
    player1Score,
    (canvas.width * 1) / 4,
    (canvas.height * 1) / 4
  );
  context.fillText(
    player2Score,
    (canvas.width * 3) / 4,
    (canvas.height * 1) / 4
  );
}

function colorCircle(centerX, centerY, radius, drawColor) {
  context.fillStyle = drawColor;
  context.beginPath();
  context.arc(centerX, centerY, radius, 0, Math.PI * 2, true);
  context.fill();
}

function colorRect(leftX, topY, width, height, drawColor) {
  context.fillStyle = drawColor;
  context.fillRect(leftX, topY, width, height);
}
