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
        self.persona = (
            "Your name is Sam, the warm and expert coordinator for Peejay Kennel. "
            "You are a dog lover first and a professional second. \n\n"
            "CONVERSATIONAL STYLE: \n"
            "1. EMPATHY FIRST: If a user shares a detail, acknowledge it with genuine warmth. If it's a puppy, show excitement; if it's a senior, show extra care. \n"
            "2. NO CHECKLISTS: Never send a list of questions. You are having a professional, friendly chat. \n"
            "3. CONSULTATIVE EXPERTISE: Use the 'Knowledge Base' to explain HOW we perform services. Offer expert advice on grooming or boarding based on the dog's needs. \n"
            "4. ONE-BY-ONE EXTRACTION: Look at 'DATA GATHERED SO FAR'. Only ask for the ONE most important missing detail next, woven naturally into a helpful suggestion. \n"
            "5. TONE: Warm, authoritative, empathetic, and boutique. Avoid robotic phrases like 'Please provide' or 'I need'."
        )

    def get_answer(self, user_query, user_info=None, history="", current_time=""):
        # Retrieve context from your knowledge_base.txt for consultations
        facts = self.vector_db.similarity_search(user_query, k=3)
        context = " ".join([f.page_content for f in facts])
        
        # Contextual summary for Sam's brain
        is_returning = "RETURNING" if user_info and user_info.get('dog_name') else "NEW LEAD"
        client_bio = f"Current Knowledge: {user_info}. History: {history}"

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"{self.persona}\n\nToday's Date: {current_time}\n\nKennel Knowledge: {context}"},
                {"role": "system", "content": f"DATA GATHERED SO FAR: {client_bio}"},
                {"role": "user", "content": (
                    f"The user said: '{user_query}'. \n\n"
                    "Action: 1. Validate their message with empathy. 2. Provide a professional 'Consultant' tip from the Knowledge Base. "
                    "3. Naturally ask for ONE missing detail to move the booking forward."
                )}
            ],
            temperature=0.7 # Higher temperature for more natural, varied speech
        )
        return response.choices[0].message.content