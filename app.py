import pandas as pd
import PyPDF2
import streamlit as st
from PIL import Image

from io import BytesIO

from dotenv import load_dotenv
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Loading Image using PIL
im = Image.open('./images/icon.jpeg')

# Set Main configuration of page (title, icon etc.)
st.set_page_config(layout="wide", page_title='AI PaperPal', page_icon=im)


hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """

st.markdown(hide_default_format, unsafe_allow_html=True)
st.markdown(
    "<h2 style='text-align: center; color: Gray;'> AI PaperPal : Your Document's All-Knowing Assistant</h2>",
    unsafe_allow_html=True)
st.text("")
st.text("")

load_dotenv()


@st.cache_data()
def extract_text_from_pdfs(pdf_files):
    # Create an empty data frame
    df = pd.DataFrame(columns=["file", "text"])
    # Iterate over the PDF files
    for pdf_file in pdf_files:
        # Open the PDF file
        # with open(pdf_file.read(), "rb") as f:
        with BytesIO(pdf_file.read()) as f:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(f)
            # Get the number of pages in the PDF
            num_pages = len(pdf_reader.pages)
            # Initialize a string to store the text from the PDF
            text = ""
            # Iterate over all the pages
            for page_num in range(num_pages):
                # Get the page object
                page = pdf_reader.pages[page_num]
                # Extract the text from the page
                page_text = page.extract_text()
                # Add the page text to the overall text
                text += page_text
            # Add the file name and the text to the data frame
            df = df.append(
                {"file": pdf_file.name, "text": text}, ignore_index=True)
    # Return the data frame
    return df


# For Alignment
col1, col2, col3 = st.columns([1, 1, 1])

# Read API Key for all users
openai_key = col1.text_input("Enter your OPENAI API Key : ", type="password")


if openai_key != "":

    # Upload the required document
    pdf_files = col3.file_uploader(
        "Upload the document (**PDF**) you need help with:",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed")

    persist_directory = 'db'

    # Once file is uploaded, start processing
    if pdf_files:
        with st.spinner("Processing PDF ..."):
            # Convert to text file and save
            df = extract_text_from_pdfs(pdf_files)

            text_splitter = CharacterTextSplitter(
                chunk_size=1000, chunk_overlap=0)
            split_texts = text_splitter.split_text(df['text'][0])

            # Get Embeddings from Open AI
            embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
            vector_db = Chroma.from_texts(
                texts=split_texts,
                embeddings=embeddings,
                persist_directory=persist_directory)

            # Get Chat Model from Open AI
            chat_model = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.2,
                openai_api_key=openai_key)
            qa = RetrievalQA.from_chain_type(
                llm=chat_model,
                chain_type="stuff",
                retriever=vector_db.as_retriever())

        # Create a chat interface
        prompt = st.chat_input("Enter your questions here...")
        if prompt:
            st.write(f"Question : {prompt}")

            with st.chat_message("AI"):
                bot_response = qa.run(prompt)
                # bot_response = "Gotcha!"
                st.write(bot_response)
