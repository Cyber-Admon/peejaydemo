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
            "Your name is Sam, the professional coordinator for Peejay Kennel. "
            "You are an expert at dog breeding and grooming logistics. "
            "GOALS: \n"
            "1. Identify if a client is new or returning. \n"
            "2. For NEW clients, politely collect: Owner Name, Dog Name, Breed, Age, Size, Vaccination Status, Health Issues, Service, and Preferred Time. \n"
            "3. For RETURNING clients, greet them and their dog by name, skip basic questions, and offer a quick rebooking. \n"
            "4. FILTER: Decline aggressive dogs over 10kg or unvaccinated dogs (explain policy). \n"
            "5. UPSSELL: Subtly suggest nail trims, de-shedding, or puppy packages if relevant. \n"
            "6. TONE: Calm, professional, and helpful."
        )

    def get_answer(self, user_query, user_info=None, history="", current_time=""):
        # RAG Retrieval
        facts = self.vector_db.similarity_search(user_query, k=2)
        context = " ".join([f.page_content for f in facts])
        
        # Contextual Awareness (Time & Client status)
        is_returning = "RETURNING CLIENT" if user_info and user_info.get('dog_name') else "NEW LEAD"
        client_bio = f"Status: {is_returning}. Info: {user_info}"

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"{self.persona}\n\nToday's Date/Time: {current_time}\n\nFacts: {context}\n\n{client_bio}"},
                {"role": "system", "content": f"Previous Chat History:\n{history}"},
                {"role": "user", "content": user_query}
            ],
            temperature=0.4 # Kept low for high professional accuracy
        )
        return response.choices[0].message.content