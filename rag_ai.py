from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.evaluation import load_evaluator
import os
import shutil
from dotenv import load_dotenv
import os
import argparse
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI



DATA_PATH = "data/"
CHROMA_PATH = "chroma"
load_dotenv()
API_KEY = os.getenv("OPEN_API_KEY")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap = 500,
        length_function=len,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(documents)
    # print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    # print(document.page_content)
    # print(document.metadata)

    return chunks


def save_to_chroma(chunks):
    # Clear out vector database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    # chroma is our vector database that we will be loading the chunks into
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(api_key=API_KEY), persist_directory=CHROMA_PATH
    )

    # db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def get_response():
    # generate_data_store()
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # prepare the DB
    embedding_function = OpenAIEmbeddings(api_key=API_KEY)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB. we pass in our query, and the number of snippets from the most relevant chunk. 
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    # if there are no relevant chunks or snippets of text that match our query, then just return so we dont continue processing.
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI(api_key=API_KEY)
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

# main()
get_response()

# embedding_function = OpenAIEmbeddings(api_key=API_KEY)
# vector = embedding_function.embed_query("apple")
# print(vector)
# print(len(vector))

# this will help us find the distance between two vectors. in other words, the similarity of two vectors

# evaluator = load_evaluator("pairwise_embedding_distance", embeddings=embedding_function)
# x = evaluator.evaluate_string_pairs(prediction="apple", prediction_b="orange") # {'score': 0.13554126333631666}
# y = evaluator.evaluate_string_pairs(prediction="apple", prediction_b="beach") # {'score': 0.2024668709009093}
# z = evaluator.evaluate_string_pairs(prediction="apple", prediction_b="apple") # {'score': 1.1102230246251565e-16}
# a = evaluator.evaluate_string_pairs(prediction="apple", prediction_b="iphone") # {'score': 0.09709082173706307}

"""
Scores closer to 0 means the words are more similar, but this is not necessarily based on text similarity.
    for example, apple and iphone has a better score than apple and orange because for the first pair, 
    apple is viewed as a company that makes iphone, but the second pair just recognizes that apple and orange are both fruits.
"""

# print(x)
# print(y)
# print(z)
# print(a)

"""
embedding vectors:
    vector representations of text that capture their meaning
    in python its a list of numbers
        so we can have to texts that are each assigned coordinates in a multi-dimensional space, and if two pieces of text are similar in meaning,
        then those coordinates are also close to each other.

    to actually generate a vector for a word, then we'll need an LLM like openAI. 
"""

"""
Querying for relevant data:
    find the chunks in our db that most likely contain the answer to our quesiton.

    take a query(our question) and turn that into an embedding using that same embedding function.
    and then scan through our db and find a few chunks that are closest in embedding distance to our query.

    from that, we can put that together and have the AI read all of that info and then decide the response to give to the user.
    
    we are using the chunks to craft a more custom response.

"""



"""
Let's try different data types too,
Let's try using data from a db.
Let's try displaying the data on our website !!


Flow for RAG

    Convert data source into chunks -> save these chunks into the Chroma DB as vectors.
    figure out the query we want to ask the db -> convert the query to a vector -> find the snippets from the db that have the closest euclidean distance to 
        our query -> Use OpenAI to derive our answer from those snippets.
"""