
import streamlit as st
import os
import json
import pyttsx3
from langchain_community.document_loaders import TextLoader
from functions import wrap_text_preserve_new_line, get_files_in_folder, model, ask_question, save_uploaded_file

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize pyttsx3 engine
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

# Inject custom CSS for styling the sidebar
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        font-size: 12px; /* Adjust the size here */
    }
    .sidebar .sidebar-content h1 {
        font-size: 16px; /* Adjust the header size here */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar with file uploader
uploaded_file = st.sidebar.file_uploader("Upload file", type=['txt'],)

if uploaded_file is not None:
    # Save the file
    file_path = save_uploaded_file(uploaded_file)

    if not isinstance (file_path,str):
        st.error("File is not uploaded")
    else:
        st.sidebar.success(f"File saved successfully at {file_path}",icon="✅")

# Get list of files and strip folder path
def update_file_list():
    return [os.path.basename(f) for f in get_files_in_folder('Document')]

files = update_file_list()

# Display file selector with placeholder
files_placeholder = ["Select a document"] + files
selected_file = st.sidebar.selectbox("Select a document", files_placeholder)

# Button to delete the selected file
if selected_file != "Select a document" and selected_file:
    if st.sidebar.button("Delete selected document"):
        try:
            file_path = os.path.join('Document', selected_file)
            os.remove(file_path)
            st.sidebar.success(f"File {selected_file} deleted successfully.",icon="✅")
            files = update_file_list()  # Update file list
            files_placeholder = ["Select a document"] + files  # Update file selector options
            selected_file = "Select a document"  # Clear selected file
        except Exception as e:
            st.sidebar.error(f"Error deleting file: {e}")

# Buttons for selecting target languages
language_options = ["English", "Gujarati", "Hindi", "Tamil"]
selected_language = st.sidebar.selectbox("Select language:", language_options)

# Streamlit UI
st.title("AI Agent For Employee Policies")

# Input for user query
query_text = st.text_input("Enter your question:")

# Determine the target language code based on user selection
language_codes = {
    "Gujarati": "gu",
    "Hindi": "hi",
    "English": "en",
    "Tamil": "ta"
}
tgt_lang = language_codes.get(selected_language, "en")


# Button to trigger the question answering
if st.button("Ask"):
    if query_text:
        ans_path = os.path.join('answer.json')
        with open(ans_path,mode = 'r') as js:
            ans_dict = json.load(js)

        if selected_file == "Select a document" :
            st.error("Please select a document to proceed.")

        elif (len(ans_dict)==0) or ((list(ans_dict.keys())[1]!=query_text) and ans_dict['doc']==selected_file) or ((list(ans_dict.keys())[1]==query_text) and ans_dict['doc']!=selected_file):
            # Load document using TextLoader if file is selected
            document = None
            if selected_file != "Select a document" and selected_file:
                st.sidebar.text(f"Selected document: {selected_file}")
                loader = TextLoader(os.path.join('Document', selected_file), encoding='utf-8')
                document = loader.load()
                # Model perform Question Answering operation
                chain = model(document) if document else None

                answer_dict = ask_question(chain, query_text, tgt_lang)

                que_ans_dict = {'doc':selected_file,query_text:answer_dict}
                
                with open(ans_path, 'w') as f:
                    json.dump(que_ans_dict, f)

                with open(ans_path,mode = 'r') as js:
                    ans_dict = json.load(js)
                lang = language_codes[selected_language]
                res_ans = ans_dict[query_text][lang]
                st.write(wrap_text_preserve_new_line(res_ans))
                
        elif (ans_dict['doc']==selected_file) and (list(ans_dict.keys())[1]== query_text):
            lang = language_codes[selected_language]
            res_ans = ans_dict[query_text][lang]
            st.write(wrap_text_preserve_new_line(res_ans))

        

