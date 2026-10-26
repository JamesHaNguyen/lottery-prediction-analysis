
const gameButtons = document.querySelectorAll(".game-button");
const modelFilter = document.querySelector("#model-filter");
const predictionList = document.querySelector("#prediction-list");
const emptyMessage = document.querySelector("#empty-message");
const errorMessage = document.querySelector("#error-message");

let predictionData = null;
let selectedGame = "Powerball";

function formatDate(dateString) {
  if (!dateString) return "Unavailable";

  // Use midday to avoid timezone shifting the displayed calendar date.
  const date = new Date(`${dateString.slice(0, 10)}T12:00:00`);

  if (Number.isNaN(date.getTime())) return dateString;

  return date.toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

function formatTimestamp(timestamp) {
  if (!timestamp) return "Unavailable";

  const date = new Date(timestamp);

  if (Number.isNaN(date.getTime())) return "Unavailable";

  return date.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function getModelType(modelName) {
  const name = modelName.toLowerCase();

  if (name.includes("frequency")) return "frequency";
  if (name.includes("random")) return "random";
  if (name.includes("markov")) return "markov";

  return "other";
}

function createBall(number, isSpecial, gameName) {
  const ball = document.createElement("span");

  ball.className = "ball";

  if (isSpecial) {
    ball.classList.add("special");

    if (gameName === "Mega Millions") {
      ball.classList.add("mega");
    }
  }

  ball.textContent = String(number).padStart(2, "0");
  ball.setAttribute(
    "aria-label",
    isSpecial
      ? `Special ball ${number}`
      : `White ball ${number}`
  );

  return ball;
}

function createPredictionCard(prediction, gameName) {
  const card = document.createElement("article");
  card.className = "prediction-card";

  const heading = document.createElement("div");
  heading.className = "card-heading";

  const title = document.createElement("h3");
  title.textContent = prediction.model;

  const tag = document.createElement("span");
  const modelType = getModelType(prediction.model);
  tag.className = `model-tag ${modelType}`;
  tag.textContent = modelType === "frequency"
    ? "Frequency"
    : modelType === "random"
      ? "Control"
      : modelType === "markov"
        ? "Markov"
        : "Model";

  heading.append(title, tag);

  const balls = document.createElement("div");
  balls.className = "ball-row";

  prediction.white_balls.forEach((number) => {
    balls.appendChild(createBall(number, false, gameName));
  });

  balls.appendChild(
    createBall(prediction.special_ball, true, gameName)
  );

  card.append(heading, balls);

  return card;
}

function renderDashboard() {
  if (!predictionData) return;

  const gameData = predictionData.games[selectedGame];

  if (!gameData) {
    predictionList.replaceChildren();
    errorMessage.hidden = false;
    return;
  }

  errorMessage.hidden = true;

  document.querySelector("#selected-game").textContent = selectedGame;
  document.querySelector("#target-date").textContent =
    formatDate(gameData.target_draw_date);

  document.querySelector("#history-date").textContent =
    formatDate(gameData.latest_history_date);

  document.querySelector("#generated-at").textContent =
    formatTimestamp(predictionData.generated_at);

  const filter = modelFilter.value;

  const filteredPredictions = gameData.predictions.filter(
    (prediction) => filter === "all" ||
      getModelType(prediction.model) === filter
  );

  document.querySelector("#model-count").textContent =
    `${filteredPredictions.length} of ${gameData.predictions.length}`;

  predictionList.replaceChildren();

  filteredPredictions.forEach((prediction) => {
    predictionList.appendChild(
      createPredictionCard(prediction, selectedGame)
    );
  });

  emptyMessage.hidden = filteredPredictions.length !== 0;
}

gameButtons.forEach((button) => {
  button.addEventListener("click", () => {
    selectedGame = button.dataset.game;

    gameButtons.forEach((otherButton) => {
      const isSelected = otherButton === button;
      otherButton.classList.toggle("active", isSelected);
      otherButton.setAttribute("aria-pressed", String(isSelected));
    });

    renderDashboard();
  });
});

modelFilter.addEventListener("change", renderDashboard);

async function loadPredictions() {
  try {
    // Avoid displaying a stale cached copy after a new deployment.
    const response = await fetch(
      `./data/predictions.json?t=${Date.now()}`
    );

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    predictionData = await response.json();

    if (!predictionData.games ||
        !predictionData.games["Powerball"] ||
        !predictionData.games["Mega Millions"]) {
      throw new Error("Prediction JSON has an unexpected format.");
    }

    renderDashboard();
  } catch (error) {
    console.error("Could not load predictions:", error);
    errorMessage.hidden = false;
    predictionList.replaceChildren();
    predictionList.textContent =
      "Prediction data is not available yet. Try again later.";
  }
}

loadPredictions();
