import os
from datetime import datetime
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from salon_logic import PeejayBot
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
bot = PeejayBot("knowledge_base.txt")

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

SESSIONS = {}

@app.route("/whatsapp", methods=['POST'])
def whatsapp_webhook():
    incoming_msg = request.values.get('Body', '')
    sender = request.values.get('From', '')
    
    # 1. Get current time for time perception
    now = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    
    # 2. Identify Client
    client_data = CLIENT_DB.get(sender, {"name": "Unknown"})
    
    # 3. Retrieve session history
    history = SESSIONS.get(sender, "")
    
    # 4. Get Sam's professional response
    reply_text = bot.get_answer(incoming_msg, client_data, history, now)
    
    # 5. Urgent/Safety Filter
    if any(word in incoming_msg.lower() for word in ["aggressive", "emergency", "biting"]):
        reply_text = "I've noted the urgency. For safety/medical reasons, I am escalating this to our manager immediately. They will contact you shortly."

    # 6. Save Session
    SESSIONS[sender] = (history + f"User: {incoming_msg}\nSam: {reply_text}\n")[-1500:]

    resp = MessagingResponse()
    resp.message(reply_text)
    return str(resp)

if __name__ == "__main__":
    app.run(port=4000, debug=True)