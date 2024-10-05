from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
import re
import streamlit as st 



google_key = st.secrets["GOOGLE_API_KEY"]
groq_key = st.secrets["GROQ_API_KEY"]

# Reading the text file
with open("fetchedata.txt", "r", encoding='latin1') as file:
    entire_text = file.read()

seperator = "_____________________________________________________________"
courses = entire_text.split(seperator)[:-1]
courses = [x.lower() for x in courses]
course_name_list = []

pattern = "course name:\\s*(.*?)\\s*\\n"
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n"])
all_docs = []

# Course detail
course_details = {}
for course in courses:
    course_name = re.findall(string=course, pattern=pattern)[0]
    course_name_list.append(course_name)
    course_details[course_name] = course

    chunks = splitter.split_text(course)
    # Creating documents
    for chunk in chunks:
        doc = Document(page_content=chunk, metadata={"course_name": course_name})
        all_docs.append(doc)


# Vector db
tokenizer = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_key)
vector_db = FAISS.from_documents(documents=all_docs, embedding=tokenizer)
vector_db.save_local("vector_db")

# LLM instance
llm = ChatGroq(name="gemma2-9b-it", api_key=groq_key)

class State(TypedDict):
    messages: Annotated[list, add_messages]

class Workflow:
    def __init__(self, llm, db_path, system_msg):
        self.model = llm
        self.vector_db = db_path
        self.system_prompt = system_msg

        self.graph_builder = StateGraph(State)
        self.graph_builder.add_node("refiner_node", self.query_refiner)
        self.graph_builder.add_node("chatbot", self.bot)
        self.graph_builder.add_node("retrieval_node", self.retrieval_node)

        self.graph_builder.add_edge(START, "refiner_node")
        self.graph_builder.add_edge("refiner_node", "retrieval_node")
        self.graph_builder.add_edge("retrieval_node", "chatbot")
        self.graph_builder.add_edge("chatbot", END)

        self.graph = self.graph_builder.compile()

    def query_refiner(self, state: State):
        user_search = state["messages"][-1].content
        prompt = f'''
        Refine this user-query so that correct retrieval from a vector database can be given to the user.
        Return only the query. No additional string should be there. The user-query is: {user_search}

        For Example: best AI course
        response: Best course to learn AI or Generative AI
        '''

        refined_user_query = self.model.invoke(prompt).content
        state["messages"].append(HumanMessage(content=refined_user_query))

    def bot(self, state: State):
        last_message = state["messages"][-1].content
        prompt = f'''
        You are a helpful AI-assistant. You have to act as a recommendation system. 
        Provide the course name as it is without altering the names. 
        If the recommendation list is empty, simply tell them to try with a different search query. 
        But do not provide anything out of the given list. 
        Here are the recommended courses: {last_message}
        '''
        ai_response = llm.invoke(prompt)
        return {"messages": [ai_response]}

    def retrieval_node(self, state: State):
        """
        This tool retrieves the most relevant courses from the course information stored in a vector database, based on the user search query.
        """
        question = state["messages"][-1].content

        if self.vector_db != "":
            db = FAISS.load_local(self.vector_db, embeddings=tokenizer, allow_dangerous_deserialization=True)

            thresh = 0.75
            res = db.similarity_search_with_score(question, k=195)

            recommendations = []
            for doc, score in res:
                if score <= thresh:
                    course_name = doc.metadata["course_name"]
                    if course_name not in recommendations:
                        recommendations.append(course_name)

        prompt = f'''
        Below are some course recommendations from Analyticsvidhya.com, based on your search:
        {str(recommendations)}
        '''

        ai_response = AIMessage(content=prompt)
        return {"messages": [ai_response]}  # Ensure to return "messages"

    def get_state(self) -> State:
        return self.state

