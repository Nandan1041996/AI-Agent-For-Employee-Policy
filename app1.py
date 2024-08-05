from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
import os
import json
import secrets
from langchain_community.document_loaders import TextLoader
from functions_1 import wrap_text_preserve_new_line, get_files_in_folder, model, ask_question
from langchain_core.documents import Document
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Generate a random secret key
app = Flask(__name__)
app.secret_key = secrets.token_hex(24)  # Required for flash messages

# Upload folder configuration
UPLOAD_FOLDER = 'Document/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    """Render the homepage with a list of text files."""
    files = [os.path.basename(f) for f in get_files_in_folder(app.config['UPLOAD_FOLDER']) if f.endswith('.txt')]
    return render_template('index.html', files=files)

@app.route('/uploads', methods=['POST'])
def upload_file():
    """Handle file uploads."""
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    if file and file.filename.lower().endswith('.txt'):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('File uploaded successfully')
    else:
        flash('Invalid file type. Only .txt files are allowed.')
    
    return redirect(url_for('index'))

@app.route('/delete/<filename>')
def delete_file(filename):
    """Delete a file."""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        flash('File deleted successfully.')
    else:
        flash('File not found.')
    
    return redirect(url_for('index'))


@app.route('/ask', methods=['POST'])
def ask_question_route():
    """Handle question-asking and return the answer."""
    query_text = request.form.get('query_text')
    selected_file = request.form.get('selected_file')
    selected_language = request.form.get('selected_language')

    print('query_text :', query_text)
    print('selected_file:', selected_file)
    print('selected_language:', selected_language)
    
    if query_text:
        print('query_text',query_text)
        ans_path = 'answer.json'
        
        # Load existing answers if they exist
        if os.path.exists(ans_path):
            with open(ans_path, 'r') as js:
                ans_dict = json.load(js)
    
        if not selected_file or selected_file == "Select a document":
            flash("Please select a document to proceed.")
            return redirect(url_for('index'))

        # Determine if the answer needs to be updated
        if (len(ans_dict) == 0) or \
           ((list(ans_dict.keys())[1] != query_text) and ans_dict.get('doc') == selected_file) or \
           ((list(ans_dict.keys())[1] == query_text) and ans_dict.get('doc') != selected_file):
           
            # Load document using TextLoader if file is selected
            document = None
            if selected_file:
                loader = TextLoader(os.path.join('Document', selected_file), encoding='utf-8')
                document = loader.load()
        
                # Pass the text and source name to the model
                chain = model(document) if document else None
                print('chain:',chain)

                # Get the answer using the model
                answer_dict = ask_question(chain, query_text, selected_language)
                print('answer_dict',answer_dict)
                # Prepare the answer dictionary
                que_ans_dict = {'doc': selected_file, query_text: answer_dict}
                
                # Save the answer dictionary to a file
                with open(ans_path, 'w') as f:
                    json.dump(que_ans_dict, f)
                # Reload the answers
                with open(ans_path, 'r') as js:
                    ans_dict = json.load(js)
                res_ans = ans_dict[query_text][selected_language]
            return jsonify({'answer': res_ans})
        elif (ans_dict['doc']==selected_file) and (list(ans_dict.keys())[1]== query_text):
            res_ans = ans_dict[query_text][selected_language]
            return jsonify({'answer': res_ans})
    else:
        res_ans = 'Answer Not Found'
        return jsonify({'answer': res_ans})
    

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)

