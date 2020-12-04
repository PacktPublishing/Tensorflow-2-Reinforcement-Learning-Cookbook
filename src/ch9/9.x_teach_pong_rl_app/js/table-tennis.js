const WINNER_SCORE = 10;
const PADDLE_HEIGHT = 100;
const PADDLE_THICKNESS = 10;

var canvas,
  context,
  ballX = 50,
  ballSpeedX = 15,
  ballY = 10,
  ballSpeedY = 4,
  paddleLeftY = 250,
  paddleRightY = 250,
  player1Score = 0,
  player2Score = 0,
  gameOver = false;

function calculateMousePos(e) {
  var rect = canvas.getBoundingClientRect(),
    root = document.documentElement;
  mouseX = e.clientX - rect.left - root.scrollLeft;
  mouseY = e.clientY - rect.top - root.scrollTop;
  // console.log(e, rect, root);
  return {
    x: mouseX,
    y: mouseY,
  };
}

function handleMouseClick(e) {
  if (gameOver) {
    player1Score = 0;
    player2Score = 0;
    gameOver = false;
  }
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

  var gameCanvas = document.getElementById("tableCanvas");
  gameCanvas.width = newWidth;
  gameCanvas.height = newHeight;
}

window.onload = function () {
  canvas = document.getElementById("tableCanvas");
  context = canvas.getContext("2d");

  // Resize Table Canvas
  resizeTable();

  var framesPerSecond = 30;
  setInterval(function () {
    moveEverything();
    drawEverything();
  }, 1000 / framesPerSecond);

  canvas.addEventListener("mousedown", handleMouseClick);
  canvas.addEventListener("mousemove", function (e) {
    var mousePos = calculateMousePos(e);
    paddleLeftY = mousePos.y - PADDLE_HEIGHT / 2;
  });

  window.addEventListener("resize", resizeTable, false);
  window.addEventListener("orientationchange", resizeTable, false);
};

function ballReset() {
  if (player1Score >= WINNER_SCORE || player2Score >= WINNER_SCORE) {
    gameOver = true;
  }

  ballSpeedX = -ballSpeedX;
  ballX = canvas.width / 2;
  ballY = canvas.height / 2;
}

function computerMovement() {
  var paddleRightYCenter = paddleRightY + PADDLE_HEIGHT / 2;
  if (paddleRightYCenter < ballY + 35) {
    paddleRightY += 6;
  } else {
    paddleRightY -= 6;
  }
}

function moveEverything() {
  if (gameOver) {
    return;
  }

  computerMovement();
  ballY += ballSpeedY;
  ballX += ballSpeedX;

  if (ballX > canvas.width) {
    if (ballY > paddleRightY && ballY < paddleRightY + PADDLE_HEIGHT) {
      console.log(paddleRightY);
      ballSpeedX = -ballSpeedX;
      var deltaY = ballY - (paddleRightY + PADDLE_HEIGHT / 2);
      ballSpeedY = deltaY * 0.35;
    } else {
      player1Score++;
      ballReset();
    }
  }

  if (ballX < 0) {
    if (ballY > paddleLeftY && ballY < paddleLeftY + PADDLE_HEIGHT) {
      console.log(paddleLeftY);
      ballSpeedX = -ballSpeedX;
      var deltaY = ballY - (paddleLeftY + PADDLE_HEIGHT / 2);
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

function drawEverything() {
  // Draw the Table
  colorRect(0, 0, canvas.width, canvas.height, "black");

  if (gameOver) {
    context.fillStyle = "white";

    if (player1Score >= WINNER_SCORE) {
      context.fillText(
        "Left Player Won!",
        canvas.width / 2,
        (canvas.height * 1) / 4
      );
    } else if (player2Score >= WINNER_SCORE) {
      context.fillText(
        "Right Player Won!",
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

  colorRect(0, paddleLeftY, PADDLE_THICKNESS, PADDLE_HEIGHT, "white");
  colorRect(
    canvas.width - PADDLE_THICKNESS,
    paddleRightY,
    PADDLE_THICKNESS,
    PADDLE_HEIGHT,
    "white"
  );
  colorCircle(ballX, ballY, 10, "white");
  context.fillText(player1Score, 100, 100);
  context.fillText(player2Score, canvas.width - 100, 100);
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
