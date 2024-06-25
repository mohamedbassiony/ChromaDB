
import csv

with open('menu_items.csv') as file:
    lines = csv.reader(file)

    documents = []
    metadatas = []
    ids = []
    id = 1

    for i, line in enumerate(lines):
        if i == 0:
            continue

        documents.append(line[1])
        metadatas.append({"item_id": line[0]})
        ids.append(str(id))
        id += 1


import chromadb
from chromadb.utils import embedding_functions
chroma_client = chromadb.Client()

#chroma_client.delete_collection(name="my_collection")

# sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
# collection = chroma_client.create_collection(name="my_collection")

# import
import chromadb.utils.embedding_functions as embedding_functions

from dotenv import load_dotenv
import os

load_dotenv(".env")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# use directly
google_ef  = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=GOOGLE_API_KEY)
# pass documents to query for .add and .query
collection = chroma_client.create_collection(name="my_collection", embedding_function=google_ef)


collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

results = collection.query(
    query_texts=["sea food"],
    n_results=5,
    include=['documents']
)

print(results)

#client = chromadb.PersistentClient(path="vectordb")