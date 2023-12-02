from googlesearch import search
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
import sys

def get_search_results(query):
    search_results = search(query, num=3, stop=3, pause=2)
    return search_results

def scrape_webpage(url):
    try:
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        return text
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return None

def main():
#    query = input("Ask your question: ")
    query = "What did Elon Musk said to X advertisers?"
    search_results = get_search_results(query)

    scraped_texts = []
    reference_links = []

    for idx, link in enumerate(search_results, 1):
        text = scrape_webpage(link)
        if text:
            scraped_texts.append(text)
            reference_links.append(f"[{idx}] - {link}")

    # Saving scraped text in a variable
    all_scraped_text = '\n'.join(scraped_texts)

    # Saving reference links
    all_reference_links = ',\n'.join(reference_links)

    print("\nScraped Text:\n")
    print(all_scraped_text)

    print("\nReference Links:\n")
    print(all_reference_links)
    
    # Split the text into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # Create a Document object for each scraped text
    # Create a Document object for each scraped text
    documents = [Document(page_content=text) for text in scraped_texts]

    # Split the documents into chunks
    text_chunks = text_splitter.split_documents(documents)

    # Download Sentence Transformers Embedding From Hugging Face
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2',
                                    model_kwargs={'device': 'cpu'})
    
    print("embeddings created")
    
    db = FAISS.from_documents(text_chunks, embeddings)

    DB_FAISS_PATH = "perplex//db_faiss"
     
    db.save_local(DB_FAISS_PATH)
    
    '''
    llm = CTransformers(model='./model/orca-mini-3b-gguf2-q4_0.gguf',
                    model_type="llama",
                    max_new_tokens=512,
                    temperature=0.3)
    
    '''
    
    #Loading the model
    def load_model():
        # Load the locally downloaded model here
        llm = CTransformers(
            #model = './model/orca-mini-3b-gguf2-q4_0.gguf',
            model = './model/orca-mini-3b.ggmlv3.q4_0.bin',
            model_type="llama",
            max_new_tokens = 512,
            temperature = 0.5
        )
        return llm
    
    llm = load_model()
    
    print("LLM initialized")
    
    # prompt template: "Provide a 2-3 sentence answer to the query based on the following sources. Be original, concise, accurate, and helpful. Cite sources as [1] or [2] or [3] after each sentence (not just the very end) to back up your answer (Ex: Correct: [1], Correct: [2][3], Incorrect: [1, 2])."
    
    qa = ConversationalRetrievalChain.from_llm(llm,retriever=db.as_retriever(search_kwargs={'k': 3}))
    
    chat_history = []
    result = qa({"question":query, "chat_history":chat_history})
    print("Response: ", result['answer'])

if __name__ == "__main__":
    main()
