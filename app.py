import streamlit as st
from model import Workflow
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage , HumanMessage , SystemMessage


groq_key = st.secrets["GROQ_API_KEY"]
google_key = st.secrets["GOOGLE_API_KEY"]



llm = ChatGroq(name = "gemma2-9b-it" , api_key = groq_key)

st.title("Smart Course Recommender: Navigating Learning Paths")
st.markdown("<h5 >Search for best data science courses at <span style= 'color: Orange'>Analyticsvidhya.com</span></h5>",
            unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Search section" , "About"])
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    user_input = st.text_input("Search here...")

    col1 , col2 , col3 = st.columns(3)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        btn = st.button("Submit",use_container_width= True)

    if btn:
        prompt = '''
        You are a helpful AI-assistant, You have to act as a recommendation system. You will be given with a
        list of recommended course names. Provide the course name as it is without doing any alteration on the names.
        If the recommendation list is empty than simply tell them to try with a different search query.But, not provide
        anything out of the given list. 
        '''
        workflow_obj = Workflow(llm=llm, db_path="vector_db", system_msg=prompt)
        message = [HumanMessage(content={user_input})]

        ai_response = workflow_obj.graph.invoke({"messages": message})
        ai_response = ai_response["messages"][-1].content
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.write("Results:")
        st.info(ai_response)

with tab2:
            st.markdown("""
            **Objective:**  
            Developed a smart search tool that allows users to find relevant free courses on Analytics Vidhya efficiently.
            
            **Data Collection:**  
            Utilized **BeautifulSoup** and **Selenium** to scrape course data, including titles, descriptions, and curricula from the Analytics Vidhya platform.
            
            **LLM Integration:**  
            Implemented the **Google Gemma** model via **ChatGroq** for natural language processing, allowing users to input queries in natural language and receive refined search results. The model was integrated using **LangGraph** to manage the interaction flow and logic between components.
            """)
            
            # Add image showing architecture
            st.image('architecture.jpeg', caption='System Architecture')
            
            st.markdown("""
            **Search System:**  
            Built a robust search mechanism that leverages a vector database to retrieve the most relevant courses based on user queries.
            
            **Deployment:**  
            Deployed the application using **Streamlit** for public accessibility, ensuring ease of use and interaction.
            """)
