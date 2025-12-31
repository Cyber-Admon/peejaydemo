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
        # Using Recursive splitter for better natural language flow
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        
        self.vector_db = Chroma.from_documents(
            documents=text_splitter.split_documents(docs), 
            embedding=OpenAIEmbeddings(),
            persist_directory="./db_peejay"
        )
        self.client = OpenAI()
        
        # Sam: The Empathetic Consultant Persona
        self.persona = (
            "Your name is Sam, the warm and expert coordinator for Peejay Kennel. "
            "You are a dog lover first and a professional second. \n\n"
            "CONVERSATION RULES: \n"
            "1. EMPATHY FIRST: Acknowledge user details with warmth. If they mention their dog, celebrate them! \n"
            "2. NO WALLS OF TEXT: Never send a list of questions. Aim for short, natural messages. \n"
            "3. CONSULTATIVE ADVICE: Provide value first. Explain your processes before asking for data. \n"
            "4. ONE-PIECE EXTRACTION: Ask for only ONE missing detail at a time. \n"
            "5. TONE: Human, deeply caring, and boutique."
        )

    # THIS IS THE MISSING METHOD CAUSING THE ERROR
    def get_answer(self, user_query, user_info=None, history="", current_time=""):
        # Retrieve context from your knowledge_base.txt
        facts = self.vector_db.similarity_search(user_query, k=3)
        context = " ".join([f.page_content for f in facts])
        
        client_bio = f"DATA GATHERED: {user_info}. HISTORY: {history}"

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"{self.persona}\n\nToday: {current_time}\nFacts: {context}"},
                {"role": "system", "content": client_bio},
                {"role": "user", "content": (
                    f"User said: '{user_query}'. \n\n"
                    "Action: Validate with empathy, give an expert tip, and ask for ONE missing detail."
                )}
            ],
            temperature=0.7 
        )
        return response.choices[0].message.content