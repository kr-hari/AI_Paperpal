from typing import Dict

import pandas as pd
import PyPDF2
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, BertForQuestionAnswering, pipeline

from io import BytesIO



import os
import sys
import time

import base64
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI




from PIL import Image
# Loading Image using PIL
im = Image.open('./images/icon.jpeg')

st.set_page_config(layout="wide", page_title = 'AI PaperPal', page_icon=im )

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """

st.markdown(hide_default_format, unsafe_allow_html=True)
# st.title("AI PaperPal : Your Document's All-Knowing Assistant :sunglasses:")
# st.markdown()
st.markdown("<h2 style='text-align: center; color: Gray;'> AI PaperPal : Your Document's All-Knowing Assistant</h2>", unsafe_allow_html=True)
# st.markdown("<h5 style='text-align: center; '>Your personal document assistant. Upload your document and ask any question you want answer to : </h5>", unsafe_allow_html=True)
st.text("")
st.text("")


load_dotenv()

# app = FastAPI()
# api_key = os.getenv("OPENAI_API_KEY")



# @st.cache_data(allow_output_mutation=True)
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
            df = df.append({"file": pdf_file.name, "text": text}, ignore_index=True)
    # Return the data frame
    return df


# def preprocess_text(text_list):
#     # Initialize a empty list to store the pre-processed text
#     processed_text = []
#     # Iterate over the text in the list
#     for text in text_list:
#         num_words = len(text.split(" "))
#         if num_words > 10:  # only include sentences with length >10
#             processed_text.append(text)
#     # Return the pre-processed text
#     return processed_text


# def remove_short_sentences(df):
#     df["sentences"] = df["sentences"].apply(preprocess_text)
#     return df


# @st.cache(allow_output_mutation=True)
# def get_relevant_texts(df, topic):
#     model_embedding = SentenceTransformer("all-MiniLM-L6-v2")
#     model_embedding.save("all-MiniLM-L6-v2")
#     cosine_threshold = 0.3  # set threshold for cosine similarity value
#     queries = topic  # search query
#     results = []
#     for i, document in enumerate(df["sentences"]):
#         sentence_embeddings = model_embedding.encode(document)
#         query_embedding = model_embedding.encode(queries)
#         for j, sentence_embedding in enumerate(sentence_embeddings):
#             distance = cosine_similarity(
#                 sentence_embedding.reshape((1, -1)), query_embedding.reshape((1, -1))
#             )[0][0]
#             sentence = df["sentences"].iloc[i][j]
#             results += [(i, sentence, distance)]
#     results = sorted(results, key=lambda x: x[2], reverse=True)
#     del model_embedding

#     texts = []
#     for idx, sentence, distance in results:
#         if distance > cosine_threshold:
#             text = sentence
#             texts.append(text)
#     # turn the list to string
#     context = "".join(texts)
#     return context

col1, col2, col3 = st.columns([1,1,1])


openai_key = col1.text_input("Enter your OPENAI API Key : ", type="password")

if openai_key != "":

    pdf_files = col3.file_uploader(
        "Upload the document (**PDF**) you need help with:", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed"
    )

    persist_directory = 'db'


    if pdf_files:
        with st.spinner("Processing PDF ..."):
            # Convert to text file and save

            # base64_pdf = base64.b64encode(pdf_files[0]).decode('utf-8')
            # pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
            # st.markdown(pdf_display, unsafe_allow_html=True)
            df = extract_text_from_pdfs(pdf_files)
            filename = "document" + time.strftime("%Y%m%d-%H%M%S")
            file_path = os.path.join('./docs', filename+".txt")
            with open(file_path, 'w') as file:
                file.write(df['text'][0])
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            loader = UnstructuredFileLoader(file_path)
            documents = loader.load()
            split_texts = text_splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
            vector_db = Chroma.from_documents(documents=split_texts, embeddings=embeddings, persist_directory=persist_directory)

            chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, openai_api_key=openai_key)
            qa = RetrievalQA.from_chain_type(llm=chat_model, chain_type="stuff", retriever=vector_db.as_retriever())

        # pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

        prompt = st.chat_input("Enter your questions here...")
        if prompt:
            st.write(f"Question : {prompt}")

            with st.chat_message("AI"):
                # bot_response = qa.run(user_input)
                bot_response = "Gotcha!"
                st.write(bot_response)
                # st.line_chart(np.random.randn(30, 3))
        # user_input = st.text_input("Enter your questions here...")

        # if user_input != "":
        #     with st.spinner("Be right back..."):
        #         bot_response = qa.run(user_input)
        #     st.write(bot_response)
