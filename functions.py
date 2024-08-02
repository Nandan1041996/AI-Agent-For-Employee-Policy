
import os
import textwrap
import pyttsx3
import streamlit as st
# from langchain.globals import set_debug
from deep_translator import GoogleTranslator
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain

# set_debug(True)
# Define text wrapping function
def wrap_text_preserve_new_line(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text 

def save_uploaded_file(uploaded_file):
    try:
        doc_path = os.path.join('Document')
        # Define the path where the file will be saved
        file_path = os.path.join(doc_path, uploaded_file.name)
        # Save the file to the specified path
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except:
        return int(1)

# Function to get all files in a directory
def get_files_in_folder(folder_path):
    files = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):  # Only consider .txt files
            files.append(os.path.join(folder_path, filename))
    return files

def ask_question(chain,query_text,tgt_lang=None):
    result = chain.invoke({'question': query_text}, return_only_outputs=True)
    answer = result['answer']
    res_dict = {'en':answer}
    # To translate
    tgt_lang_lst = ['gu','hi','ta']
    for tgt_lang in tgt_lang_lst:
        translated = GoogleTranslator(source='en', target = tgt_lang).translate(answer)
        res =wrap_text_preserve_new_line(translated)
        res_dict[tgt_lang] = res
    return res_dict

def model(document):
    # Initialize RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n','\n','.'],chunk_size=1500, chunk_overlap=300)
    docs = text_splitter.split_documents(document)
    # Initialize embeddings and vector index
    embeddings = HuggingFaceBgeEmbeddings()
    vector_index = FAISS.from_documents(docs, embeddings)

    # Initialize Hugging Face LLM (Language Model) Endpoint
    llm = HuggingFaceEndpoint(repo_id='mistralai/Mistral-7B-Instruct-v0.3', temperature=0.5, token='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', max_new_tokens=1000)
    # Initialize Question Answering Chain
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_index.as_retriever())
    return chain

# Function to speak the answer
def speak(answer_dict):
    if answer_dict['lang'] == 'English' and 'answer' in answer_dict:
        engine = pyttsx3.init('sapi5')
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)
        engine.say(answer_dict['answer'])
        engine.runAndWait()

# Function to play audio
def play_audio(answer):
    with st.spinner("Loading audio..."):
        speak(answer)