import os
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PeejayBot:
    def __init__(self, data_path):
        # Recursive splitting keeps sentences together for more human-like "facts"
        loader = TextLoader(data_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        
        self.vector_db = Chroma.from_documents(
            documents=text_splitter.split_documents(docs), 
            embedding=OpenAIEmbeddings(),
            persist_directory="./db_peejay"
        )
        self.client = OpenAI()
        
        # Sam: The Empathetic Consultant Persona
        # Update this in salon_logic.py
        self.persona = (
            "Your name is Sam, the expert coordinator for Peejay Kennel. "
            "Your goal is to be helpful and empathetic, NOT repetitive. \n\n"
            "STRICT LOGIC RULES: \n"
            "1. CHECK HISTORY: Look at what the user just said. If they asked a question, answer it directly first. \n"
            "2. NO REPETITION: Look at the 'DATA GATHERED SO FAR'. If you already know the dog's name, breed, or any other detail, NEVER ask for it again. \n"
            "3. PROGRESSIVE GATHERING: If the user seems confused or says 'huh?', stop the sales pitch. Empathize, simplify, and ask only ONE very simple question. \n"
            "4. CONSULTATION: If you have answered a question, provide one brief expert tip (e.g., about coat health), then stop talking. Keep messages under 3 sentences. \n"
            "5. TONE: Human, concise, and professional."
        )

        def get_answer(self, user_query, user_info=None, history="", current_time=""):
            facts = self.vector_db.similarity_search(user_query, k=2)
            context = " ".join([f.page_content for f in facts])
            
            # We explicitly pass the gathered info as a "State"
            client_bio = f"ALREADY KNOWN: {user_info}. \nRECENT HISTORY: {history}"

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"{self.persona}\n\nToday's Date: {current_time}\nFacts: {context}"},
                    {"role": "system", "content": client_bio},
                    {"role": "user", "content": f"The user just said: '{user_query}'. If they are confused, explain simply. Otherwise, ask for ONE missing detail."}
                ],
                temperature=0.5 # Lowered slightly to prevent Sam from 'hallucinating' long stories
            )
            return response.choices[0].message.content