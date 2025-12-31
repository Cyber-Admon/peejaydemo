import os
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PeejayBot:
    def __init__(self, data_path):
        # Use Recursive splitter to keep sentences together for more coherent "facts"
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
            "Your name is Sam, the expert and warm coordinator for Peejay Kennel. "
            "You are a dog lover first and a professional second. \n\n"
            "CONVERSATION RULES: \n"
            "1. EMPATHY & VALIDATION: Always acknowledge user details with warmth. If they mention their dog, celebrate them! (e.g., 'A Goldendoodle! They have such wonderful personalities.') \n"
            "2. NO WALLS OF TEXT: Never send a list of questions. Aim for short, natural messages. \n"
            "3. CONSULTATIVE ADVICE: Use the 'Knowledge Base' to provide value first. Explain our grooming process or care standards before asking for information. \n"
            "4. STRATEGIC EXTRACTION: Check the 'DATA GATHERED SO FAR'. Only ask for the ONE most important missing detail next. Weave it into a helpful suggestion. \n"
            "5. TONE: Professional but deeply caring. Avoid robotic phrases like 'Please provide' or 'I need'."
        )

    def get_answer(self, user_query, user_info=None, history="", current_time=""):
        # Retrieve context from your knowledge_base.txt
        facts = self.vector_db.similarity_search(user_query, k=3)
        context = " ".join([f.page_content for f in facts])
        
        # Contextual summary for the LLM
        is_returning = "RETURNING CLIENT" if user_info and user_info.get('dog_name') else "NEW LEAD"
        client_bio = f"Status: {is_returning}. Known Info: {user_info}. Session History: {history}"

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"{self.persona}\n\nToday's Date: {current_time}\n\nKnowledge Base: {context}"},
                {"role": "system", "content": f"DATA GATHERED SO FAR: {client_bio}"},
                {"role": "user", "content": (
                    f"The user said: '{user_query}'. \n\n"
                    "Your Task: \n"
                    "1. Respond with genuine empathy and validation. \n"
                    "2. Offer a professional consultation tip from the Knowledge Base. \n"
                    "3. Naturally ask for the ONE next piece of information you need to move toward a booking."
                )}
            ],
            temperature=0.7 # Higher temperature for natural, human-like variety
        )
        return response.choices[0].message.content