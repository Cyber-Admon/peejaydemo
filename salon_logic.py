import os
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PeejayBot:
    def __init__(self, data_path):
        loader = TextLoader(data_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        
        self.vector_db = Chroma.from_documents(
            documents=text_splitter.split_documents(docs), 
            embedding=OpenAIEmbeddings(),
            persist_directory="./db_peejay"
        )
        self.client = OpenAI()
        
        self.persona = (
            "You are Sam, a professional kennel coordinator. Talk like a real person, not an AI. \n\n"
            "STRICT RULES: \n"
            "1. NO REPETITION: Look at 'KNOWN INFO'. If you already have a detail, NEVER ask for it or confirm you are 'excited to learn it' again. \n"
            "2. DIRECT ANSWERS: If the user gives you a piece of info (like a name), just acknowledge it briefly and move to the next logical question. \n"
            "3. NO UNSOLICITED ADVICE: Only give a grooming/care tip IF the user asks about a service. If they are just giving you their dog's name, don't give a tip. \n"
            "4. BREVITY: Keep responses under 40 words. \n"
            "5. EMPATHY: Be warm, but don't be 'fake' or overly enthusiastic."
        )

    def get_answer(self, user_query, user_info=None, history="", current_time=""):
        facts = self.vector_db.similarity_search(user_query, k=2)
        context = " ".join([f.page_content for f in facts])
        
        # We pass ONLY the last 3 exchanges to prevent context drift
        short_history = history[-800:]
        client_state = f"KNOWN INFO: {user_info}. \nHISTORY: {short_history}"

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"{self.persona}\n\nFacts: {context}"},
                {"role": "system", "content": client_state},
                {"role": "user", "content": (
                    f"User: '{user_query}'. \n\n"
                    "Instructions: \n"
                    "1. If info was provided, update your internal checklist. \n"
                    "2. Do NOT give a tip unless they asked about a service. \n"
                    "3. Ask for the next missing detail from the checklist (Breed, Age, Size, Vax, Health, Service, Time)."
                )}
            ],
            temperature=0.3 # Lowered to 0.3 to stop 'creative' unsolicited tips
        )
        return response.choices[0].message.content