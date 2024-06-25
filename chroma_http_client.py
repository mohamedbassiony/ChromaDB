import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

# import
import chromadb.utils.embedding_functions as embedding_functions

from dotenv import load_dotenv
import os

load_dotenv(".env") 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_SERVER_AUTHN_PROVIDER = os.getenv("CHROMA_SERVER_AUTHN_PROVIDER")
CHROMA_SERVER_AUTHN_CREDENTIALS = os.getenv("CHROMA_SERVER_AUTHN_CREDENTIALS")

# Connect with no authentication
# chroma_client = chromadb.HttpClient(host='localhost', port=8800,)

# Connect with token authentication
# chroma_client = chromadb.HttpClient(host='localhost', port=8800,
#     settings=Settings(
#         chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
#         chroma_client_auth_credentials="test-token"
#     )
# )

# Connect with role-based authentication
chroma_client = chromadb.HttpClient(host='localhost', port=8800,
    settings=Settings(
        chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
        chroma_server_authn_provider="chromadb.auth.simple_rbac_authz.SimpleRBACAuthorizationProvider",
        chroma_client_auth_credentials="test-token-readonly"
    )
)



# use directly
google_ef  = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=GOOGLE_API_KEY)
# pass documents to query for .add and .query
collection = chroma_client.get_collection(name="my_collection", embedding_function=google_ef)

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

# collection.add(
#     documents=documents,
#     metadatas=metadatas,
#     ids=ids
# )

results = collection.query(
    query_texts=["sea food"],
    n_results=5,
    include=['documents']
)

print(results)