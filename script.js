const chatbotButton = document.getElementById("chatbot-button");
const chatPanel = document.getElementById("chat-panel");
const sendBtn = document.getElementById("send-btn");
const userInput = document.getElementById("user-input");
const chatBody = document.getElementById("chat-body");

chatbotButton.addEventListener("click", () => {
  chatPanel.style.display = "flex";
});

sendBtn.addEventListener("click", async () => {
  const question = userInput.value.trim();
  if (!question) return;

  // Append user's question to chat
  appendMessage("user", question);

  // Clear input field
  userInput.value = "";

  try {
    const res = await fetch("http://localhost:5000/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const data = await res.json();
    const answer = data.answer || "Sorry, no answer available.";
    appendMessage("ai", answer);
  } catch (err) {
    appendMessage("ai", "Something went wrong while getting a response.");
  }
});

function appendMessage(sender, message) {
  const div = document.createElement("div");
  div.className = sender === "ai" ? "message ai-message" : "message user-message";
  div.textContent = message;
  chatBody.appendChild(div);
  chatBody.scrollTop = chatBody.scrollHeight;
}
