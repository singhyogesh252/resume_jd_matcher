import streamlit as st
from rag_pipeline import main, match_resume_to_jd
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv

load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.3,
    max_tokens=512
)

st.set_page_config(page_title="LLM-Powered Resume & Job Match Advisor", layout="wide")
st.title("Resume & JD Match Assistant")

st.sidebar.header("Upload Your Resume")
uploaded_file = st.sidebar.file_uploader("Choose a PDF resume", type="pdf")

st.sidebar.header("Paste Job Description")
jd_text = st.sidebar.text_area("Paste the JD here", height=300)

if uploaded_file and jd_text:
    resumes, jd_doc, jd_vectorstore = main(uploaded_file, jd_text)

    st.header("Resume–JD Match Feedback")
    feedback_prompt_template = """
    You are a smart job assistant. Use the resume and job description provided by the user to:
    1. Match skills and experience.
    2. Suggest improvements in the resume to align with the JD.
    3. List skill gaps or mismatches.

    Resume:
    {resume}

    Job Description:
    {jd}

    Based on the above, generate helpful feedback.
    """
    jd_content_string = ""
    resume_content = ""
    for resume in resumes:
            resume_content += resume.page_content
            matches = match_resume_to_jd(resume.page_content, jd_vectorstore)
            for match in matches:
                jd_content = match.page_content
                jd_content_string += jd_content


    with st.spinner("Generating feedback..."):
        
        prompt = feedback_prompt_template.format(
            resume=resume_content,
            jd=jd_content_string
        )
        feedback = llm.invoke(prompt)
        st.subheader("Matched JD Snippet:")
        st.markdown(jd_content_string[:300] + "...")
        st.subheader("Feedback:")
        st.write(feedback.content)
    
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        system_message = SystemMessagePromptTemplate.from_template(
        """
        You are a professional AI assistant specialized in resume and job description analysis.
        Use the resume, JD, and feedback provided to give tailored responses to user questions.
        Resume:
        {resume_content}
        
        Matched JD Content:
        {jd_content_string}
        
        Feedback:
        {feedback.content}
        """
    )

        # Set up a conversational chat template with memory placeholder
        prompt_template = ChatPromptTemplate.from_messages([
            system_message,
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        # st.session_state.memory.chat_memory.add_ai_message("Hello! How can I help you regarding your resume or the job description?")
        # st.session_state.memory.chat_memory.add_user_message(intro_context)

        # Create the conversation chain
        chat_chain = LLMChain(
            llm=llm,
            prompt=prompt_template,
            memory=st.session_state.memory,
            verbose=False
        )

        # UI for chat
        st.subheader("Chat with the Assistant")
        user_input = st.chat_input("Ask about resume, JD, or suggestions...")

        if user_input:
            response = chat_chain.predict(
                input=user_input,
                resume=resume.page_content,
                jd_content=jd_content_string[:500],
                feedback=feedback.content
            )
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                st.markdown(response)

elif not uploaded_file:
    st.warning("Please upload your resume (PDF).")

elif not jd_text:
    st.warning("Please paste the job description.")


