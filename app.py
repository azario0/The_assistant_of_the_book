from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import os
import sqlite3
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
import pickle


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a random secret key

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'epub'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database setup
def init_db():
    conn = sqlite3.connect('books.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS books
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  title TEXT NOT NULL,
                  filename TEXT NOT NULL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS conversations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  book_id INTEGER,
                  content TEXT,
                  memory BLOB,
                  FOREIGN KEY (book_id) REFERENCES books (id))''')
    conn.commit()
    conn.close()


init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    conn = sqlite3.connect('books.db')
    c = conn.cursor()
    c.execute("SELECT * FROM books")
    books = c.fetchall()
    conn.close()
    return render_template('index.html', books=books)

@app.route('/upload', methods=['GET', 'POST'])
def upload_book():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the uploaded book
            api_key = session.get('api_key')
            if not api_key:
                api_key = request.form['api_key']
                session['api_key'] = api_key
            process_book(filepath, api_key)
            
            # Save book info to database
            conn = sqlite3.connect('books.db')
            c = conn.cursor()
            c.execute("INSERT INTO books (title, filename) VALUES (?, ?)", (filename, filepath))
            conn.commit()
            conn.close()
            
            flash('Book uploaded successfully')
            return redirect(url_for('index'))
    return render_template('upload.html')

@app.route('/delete/<int:book_id>', methods=['POST'])
def delete_book(book_id):
    conn = sqlite3.connect('books.db')
    c = conn.cursor()
    
    # Delete the book
    c.execute("DELETE FROM books WHERE id = ?", (book_id,))
    
    # Delete associated conversations
    c.execute("DELETE FROM conversations WHERE book_id = ?", (book_id,))
    
    conn.commit()
    conn.close()
    flash('Book and its conversations deleted successfully')
    return redirect(url_for('index'))
def process_conversation(content):
    """Process the conversation content to add HTML tags."""
    content = content.replace("Q:", "<div class='question'>Q:")
    content = content.replace("A:", "</div><div class='answer'>A:")
    content += "</div>"
    return content

@app.route('/conversations/<int:book_id>')
def conversations(book_id):
    conn = sqlite3.connect('books.db')
    c = conn.cursor()
    c.execute("SELECT id, content FROM conversations WHERE book_id = ?", (book_id,))
    conversations = c.fetchall()
    c.execute("SELECT title FROM books WHERE id = ?", (book_id,))
    book_title = c.fetchone()[0]
    conn.close()
    
    processed_conversations = []
    for conv in conversations:
        first_question = conv[1].split('\n')[0]  # Get the first line (question) of the conversation
        processed_conversations.append((conv[0], first_question))
    
    return render_template('conversations.html', conversations=processed_conversations, book_id=book_id, book_title=book_title)

@app.route('/conversation/<int:book_id>/<int:conversation_id>', methods=['GET', 'POST'])
def conversation(book_id, conversation_id):
    if request.method == 'POST':
        query = request.form['query']
        api_key = session.get('api_key')
        if not api_key:
            api_key = request.form['api_key']
            session['api_key'] = api_key
        
        # Retrieve the memory from the database
        conn = sqlite3.connect('books.db')
        c = conn.cursor()
        c.execute("SELECT memory FROM conversations WHERE id = ?", (conversation_id,))
        memory_data = c.fetchone()[0]
        if memory_data:
            memory = pickle.loads(memory_data)
        else:
            memory = ConversationBufferMemory()
        
        # Initialize the conversation chain with existing memory
        chain = initialize_conversation_chain(api_key, memory)
        
        # Get the response
        response = chain.invoke({"query": query})
        
        # Append to the existing conversation and update memory
        c.execute("SELECT content FROM conversations WHERE id = ?", (conversation_id,))
        existing_content = c.fetchone()[0]
        new_content = f"{existing_content}\nQ: {query}\nA: {response['result']}"
        memory_data = pickle.dumps(memory)
        c.execute("UPDATE conversations SET content = ?, memory = ? WHERE id = ?", (new_content, memory_data, conversation_id))
        conn.commit()
        conn.close()
        
        return redirect(url_for('conversation', book_id=book_id, conversation_id=conversation_id))
    
    conn = sqlite3.connect('books.db')
    c = conn.cursor()
    c.execute("SELECT content FROM conversations WHERE id = ?", (conversation_id,))
    conversation_content = c.fetchone()[0]
    c.execute("SELECT title FROM books WHERE id = ?", (book_id,))
    book_title = c.fetchone()[0]
    conn.close()
    
    processed_content = process_conversation(conversation_content)
    
    return render_template('conversation.html', conversation=processed_content, book_id=book_id, conversation_id=conversation_id, book_title=book_title)

@app.route('/new_conversation/<int:book_id>', methods=['POST'])
def new_conversation(book_id):
    query = request.form['query']
    api_key = session.get('api_key')
    if not api_key:
        api_key = request.form['api_key']
        session['api_key'] = api_key
    
    # Initialize the conversation chain
    chain = initialize_conversation_chain(api_key)
    
    # Get the response
    response = chain.invoke({"query": query})
    
    # Save the new conversation to the database
    conn = sqlite3.connect('books.db')
    c = conn.cursor()
    c.execute("INSERT INTO conversations (book_id, content) VALUES (?, ?)", 
              (book_id, f"Q: {query}\nA: {response['result']}"))
    new_conversation_id = c.lastrowid
    conn.commit()
    conn.close()
    
    return redirect(url_for('conversation', book_id=book_id, conversation_id=new_conversation_id))

@app.route('/delete_conversation/<int:book_id>/<int:conversation_id>', methods=['POST'])
def delete_conversation(book_id, conversation_id):
    conn = sqlite3.connect('books.db')
    c = conn.cursor()
    c.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()
    flash('Conversation deleted successfully')
    return redirect(url_for('conversations', book_id=book_id))

def process_book(filepath, api_key):
    loader = UnstructuredEPubLoader(filepath)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_documents = text_splitter.split_documents(data)
    
    embeddings = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004', google_api_key=api_key)
    
    batch_size = 96
    num_batches = len(all_documents) // batch_size + (len(all_documents) % batch_size > 0)
    
    texts = ["FAISS is an important library", "LangChain supports FAISS"]
    db = FAISS.from_texts(texts, embeddings)
    retv = db.as_retriever()
    
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = (batch_num + 1) * batch_size
        batch_documents = all_documents[start_index:end_index]
        retv.add_documents(batch_documents)
        print(start_index, end_index)
    
    db.save_local("faiss_index")

def initialize_conversation_chain(api_key, memory=None):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    embeddings = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004', google_api_key=api_key)
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retv = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    template = """You are a helpful assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {question}
    Helpful Answer:"""
    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
    
    if not memory:
        memory = ConversationBufferMemory()
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retv,
        memory=memory,
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return chain


if __name__ == '__main__':
    app.run(debug=True)