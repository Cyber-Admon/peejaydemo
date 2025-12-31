import os
from datetime import datetime
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

class PeejayBot:
    def __init__(self, data_path):
        loader = TextLoader(data_path)
        docs = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        self.vector_db = Chroma.from_documents(
            documents=text_splitter.split_documents(docs), 
            embedding=OpenAIEmbeddings(),
            persist_directory="./db_peejay"
        )
        self.client = OpenAI()
        
        # Professional Sam Persona with lead qualification instructions
        self.persona = (
            "Your name is Sam, the expert coordinator for Peejay Kennel. "
            "You provide high-end grooming and boarding consultations. "
            "CONVERSATIONAL STYLE: \n"
            "1. ACTIVE LISTENING: If a user gives you information, acknowledge it with expertise (e.g., 'A Goldendoodle! They have such specific coat needs.'). \n"
            "2. CONSULTATIVE SELLING: Use the 'Facts' provided to explain HOW we perform services. If they ask for grooming, briefly explain our process (e.g., our 7-point health check). \n"
            "3. ONE-BY-ONE EXTRACTION: Never send a numbered list of questions. Look at the 'Known Info' and only ask for the ONE most important missing piece of data next. \n"
            "4. DYNAMIC FILTERING: If the conversation suggests a dog might be over 10kg and aggressive, pivot the conversation to safety policies immediately. \n"
        )

    def get_answer(self, user_query, user_info=None, history="", current_time=""):
        # RAG Retrieval for specific service details (prices, methods, etc.)
        facts = self.vector_db.similarity_search(user_query, k=3)
        context = " ".join([f.page_content for f in facts])
        
        # This bio tells Sam what is already known so he doesn't repeat questions
        is_returning = "RETURNING" if user_info and user_info.get('dog_name') else "NEW LEAD"
        client_bio = f"Current Knowledge: {user_info}. History: {history}"

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"{self.persona}\n\nToday's Date: {current_time}\n\nKnowledge Base: {context}"},
                {"role": "system", "content": f"DATA GATHERED SO FAR: {client_bio}"},
                {"role": "user", "content": f"The user just said: '{user_query}'. Respond with expert advice and ask for only ONE missing detail."}
            ],
            temperature=0.7 # Higher temperature allows for more natural, varied speech
        )
        return response.choices[0].message.content