from dotenv import load_dotenv
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# Step 1 
if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

docs_dir = Path("../data/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"
print(f"Directory exists: {docs_dir.exists()}")

# Step 2
documents = SimpleDirectoryReader("../data/groundwork_docs").load_data()
print(f"number of documents loaded: {len(documents)}")
for doc in documents:
    print(doc.metadata["file_name"])

# Step 3
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)
print("Index built successfully. Ready to answer questions.")

# Step 4
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for question in questions:
    response = query_engine.query(question)

    print("-" * 30)
    print(f"Question: {question}")
    print(f"\nAnswer: {response}")

    top_node = response.source_nodes[0]
    file_name = top_node.node.metadata.get("file_name", "Unknown")
    score = top_node.score
    chunk_text = top_node.node.get_content()[:200]

    print("\nTop Retrieved Source:")
    print(f"Document: {file_name}")
    print(f"Similarity Score: {score}")
    print(f"Chunk: {chunk_text}")
    print()

# The assistant sounded confident and the responses were mostly accurate.
    # The answers about weekend hours, Groundwork's history, and wholesale/catering
    # were well supported by the retrieved source documents. I was interested in the
    # milk answer since itm entioned the top source was seasonal_specials. While that isn't
    # too bad since it does have options for dairy free, im surprised taht the menu.txt wasn't the top source
    # since in that it explicity says "All dairy-free options are available at no extra charge." 

# Step 5
failure = "What is Groundwork Coffee's annual revenue?"
response = query_engine.query(failure)
print("-" * 30)
print(f"Question: {failure}")
print(f"\nAnswer: {response}")
for i, source_node in enumerate(response.source_nodes[:3], start=1):
    file_name = source_node.node.metadata.get("file_name", "Unknown")
    score = source_node.score
    chunk_text = source_node.node.get_content()[:200]

    print(f"\nSource Node {i}:")
    print(f"Document: {file_name}")
    print(f"Similarity Score: {score}")
    print(f"Chunk: {chunk_text}")

# I asked about revenue of the company since for our documents that isn't shared at all with the model and i was curious
    #if it wouuld make up any information. To my surprised it mentioned that "the information does not include details on
    # number of customer, average spending, total sales volume, etc..." It didn't answer the questoin and admitted to not knowing
    # which was good to see. Honestly I wouldn't change much about it since it already does the job fairly well and doesn't 
    # make up anything (i ran it a few times to get different responses and it never once made up something)

# Step 6
# 1. The manual keyword RAG required substantially more code because I had
    # to implement the retrieval process myself, including tokenizing the query,
    # removing stopwords, calculating word overlap, scoring documents, and
    # selecting the best match as shown in the lesson. With LlamaIndex, it was mostly just 3 lines and it shows
    # the importance of the framework since there are less chances of bugs but also more boilerplate to help with setup

# 2. A different use case would be a sports organization that uses a RAG
    # assistant to answer questions from coaching manuals, league rules, team
    # policies, scouting reports, and player development documents. It can be team specific
    # to look at player staistics for that team if yoy are a coach or if you are a general manager
    # that needs to know more of the financies, we can load the documents and ask questions to the bot
    # without having to manually search for that info one doc by one

# 3. One failure mode that RAG cannot fully prevent is hallucination. Even
    # when the correct information is retrieved, the language model can still
    # misunderstand, combine, or incorrectly interpret the retrieved information.
    # This means that good retrieval improves reliability but does not guarantee
    # that the final answer will always be correct and can be overconfident when
    # giving that incorrect information