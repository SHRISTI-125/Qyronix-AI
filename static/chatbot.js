document.addEventListener("DOMContentLoaded", () => {

    console.log("🔥 Qyronix Chatbot Loaded");

    // elements
    const chatFab = document.getElementById("chat-fab");
    const chatPanel = document.getElementById("chat-panel");
    const chatClose = document.getElementById("chat-close");

    const sendBtn = document.getElementById("chat-send");
    const input = document.getElementById("chat-input");
    const body = document.getElementById("chat-body");

    // open chat
    chatFab.addEventListener("click", () => {
        console.log("CHAT OPENED");
        chatPanel.style.display = "flex";
    });

    //chat closed
    chatClose.addEventListener("click", () => {
        chatPanel.style.display = "none";
    });
    
    function addMessage(text, type) {
        const msg = document.createElement("div");
        msg.className = `chat-msg ${type}`;
        msg.innerText = text;
        body.appendChild(msg);
        body.scrollTop = body.scrollHeight;
        return msg;
    }

    // send msg
    async function sendMessage() {
        const text = input.value.trim();
        if (!text) return;

        // USER MESSAGE
        addMessage(text, "user");
        input.value = "";

        // BOT THINKING
        const loading = addMessage("Thinking...", "bot");

        try {
            console.log("Sending:", text);
            const response = await fetch("/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    message: text
                })
            });
            console.log("STATUS:", response.status);
            const data = await response.json();
            console.log(" DATA:", data);

            // REMOVE THINKING
            loading.remove();

            // SHOW BOT RESPONSE
            addMessage(data.response, "bot");

        } catch (error) {
            console.error(" ERROR:", error);
            loading.remove();
            addMessage(" Server Error", "bot");
        }
    }

    // button clicked
    sendBtn.addEventListener("click", sendMessage);

    //enter key
    input.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            sendMessage();
        }
    });

});