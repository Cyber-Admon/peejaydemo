import os
import json
from datetime import datetime
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from salon_logic import PeejayBot
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# Base directory for absolute paths on VPS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
bot = PeejayBot(os.path.join(BASE_DIR, "knowledge_base.txt"))

# MOCK DATABASE (Returning clients)
CLIENT_DB = {
    "whatsapp:+1234567890": {
        "name": "Alex", 
        "dog_name": "Luna", 
        "breed": "Goldendoodle", 
        "vax": "Up to date", 
        "history": "Full Groom"
    }
}

# --- PERSISTENT MEMORY MANAGEMENT ---
def load_sessions():
    """Load chat history from a JSON file to survive server restarts."""
    file_path = os.path.join(BASE_DIR, "sessions.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return {}

def save_sessions(sessions):
    """Save the current session state to disk."""
    file_path = os.path.join(BASE_DIR, "sessions.json")
    with open(file_path, "w") as f:
        json.dump(sessions, f)

SESSIONS = load_sessions()

@app.route("/whatsapp", methods=['POST'])
def whatsapp_webhook():
    incoming_msg = request.values.get('Body', '')
    sender = request.values.get('From', '')
    
    # 1. Perception: Identify current context
    now = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    client_data = CLIENT_DB.get(sender, {"name": "New Guest"})
    
    # 2. Memory: Retrieve persistent history
    history = SESSIONS.get(sender, "")
    
    # 3. Action: Get Sam's empathetic, consultative response
    reply_text = bot.get_answer(incoming_msg, client_data, history, now)
    
    # 4. Urgent/Safety Filter (Escalation Logic)
    urgent_keywords = ["aggressive", "emergency", "biting", "blood", "hurt"]
    if any(word in incoming_msg.lower() for word in urgent_keywords):
        reply_text = (
            "I've noted the urgency regarding this. For safety and the well-being of our "
            "staff and guests, I am escalating this to our manager immediately. "
            "They will contact you shortly."
        )

    # 5. Feedback Loop: Save the interaction to memory
    # Pruning to the last 2000 characters to keep Sam focused
    new_history = history + f"\nUser: {incoming_msg}\nSam: {reply_text}"
    SESSIONS[sender] = new_history[-2000:]
    save_sessions(SESSIONS)

    resp = MessagingResponse()
    resp.message(reply_text)
    return str(resp)

if __name__ == "__main__":
    # In production on your VPS, Gunicorn will handle this.
    app.run(port=4000, debug=True)