from dotenv import load_dotenv
import os
import string
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# --- RAG Concepts ---

## Q1

# Scenario A: RAG
    # RAG is the choice here since the asisstant needs to answer from a large, interntal
    # source of information, in this case hundreds of pdfs that are updated every quarter and we can use
    # retrieval to get up to date info without having to retrain the model entirely.

# Scenario B: Fine-tuning
    # Fine-tuning is best here because the startup has 3,000 examples showing 
    # a distinctive writing style. So if we train the model on these examples, we should get results
    # that reproduce the style consistently. 

# Scenario C: Prompt engineering
    # Prompt engineering is good here because the analyst only needs to ask
    # questions about a one short, two-page report. In addition its just for this one document
    # and nothing else so its just a quick prompt question

## Q2
# A confidently wrong answer is more harmful than "I am not sure" because
    # people are more likely to trust and act on an answer that sounds certain,
    # even when the information is incorrect. For example, if an AI confidently
    # gives incorrect medical advice about a medication dosage, someone could
    # follow it and be seriously harmed. The model's tone affects trust because fluent, condifence
    # in the response makes it seem like its correct and factual when in reality its just made up garbage 
    # or flat out misleading information. 

## Q3
# 1. Extract text from source documents - text is extracted from the documents that will serve as the knowledge base.
# 2. Split text into chunks - extracted text is divided into smaller, manageable sections for retrieval.
# 3. Convert text chunks into embeddings - each text chunk is converted into a numerical embedding so it can be compared with the user's query.
# 4. Receive the user's query - the model accepts the user's question as the input to the pipeline.
# 5. Embed the user's query - the query is converted into a numerical embedding representing its meaning.
# 6. Retrieve the most relevant chunks - the system finds the document chunks whose embeddings are most similar to the query.
# 7. Inject retrieved chunks into the prompt - the relevant chunks are added to the prompt as context for the LLM.
# 8. Generate a response from the LLM
#    The LLM uses the user's query and retrieved context to produce an answer.

# --- Keyword RAG ---
def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our", 'your'
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

## Q1
query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

results = simple_keyword_retrieval(query, documents, verbose=True)
print(results[0][1])

# I got hours.txt, i added "your" to the stop word list which makes it so that hours is the only one with a keyword of "weekends"
# which is a direct word from the query that we are asking so we select that hours.txt document

## Q2
query2 = "Do you have anything without caffeine?"
results2 = simple_keyword_retrieval(query2, documents, verbose=True)
print(results2[0][1])
# Our rag fails here as no document aws selected since we are using an exact-word matching rag which means
# if we dont literally ahve the exact wording/spelling, we will miss it. We should use an embedding approach instead
# since those match better and perform better in general. 

## Q3
query3 = "How do I sign up for rewards?"
results3 = simple_keyword_retrieval(query3, documents, verbose=True)
print(results3[0][1])

# My prediction was loylaty.txt but instead it said no document was selected. Similar to the previous question where
# the limitations is that we need the exact word to be found, it doesnt pick up on words that have similar meanings like
# rewards/loyalty and sign up/join. Once again we should use embeddings instead. 

# --- Semantic RAG ---

## Q1
# 1. A vector embedding is a numerical representation of text that captures its meaning and relationships to other text.
# 2. The chunk with a cosine similarity of 0.85 is more relevant. The higher score means its meaning is more similar to the query than the chunk
#    with a score of 0.30.
# 3. Semantic search compares the meaning of the text rather than just matching exact words, so it can recognize that different words can express similar ideas.

## Q2
# | Feature                    | Keyword RAG                       | Semantic RAG |
# |----------------------------|-----------------------------------|--------------|
# | What is compared?          | Exact word overlap                | embedding similarity |
# | What is retrieved?         | Full document                     | relevant chunks |
# | Can it handle synonyms?    | No                                | yes            |
# | Storage format             | Plain text dictionary             | vector store / embeddings index |
# | Relevance score            | Number of overlapping keywords    | cosine similarity (-1 to 1) |

# --- LlamaIndex ---

## Q1
docs = SimpleDirectoryReader('../data/brightleaf_pdfs').load_data()
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=3)
questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]
for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        #print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)

# Query 1: "What employee benefits does BrightLeaf offer?"
# The retrieved chunks are mostly relevant: the benefits document has a very
# high similarity score (0.9097), although the Overview and Security chunks
# were also retrieved unexpectedly. It sounds confident and specific,
# listing many benefits without unecessary phrases or uncertainty.

# Query 2: "What are BrightLeaf's security policies?"
# The security chunk is highly relevant with a similarity score of 0.8830,
# but it mentioned benefits and general Overview information which wasnt needed.
# It also sounds confident and specific, giving detailed security policies.

## Q2
query_engine_1 = index.as_query_engine(similarity_top_k=1)
print("similarity_top_k=1")
response_1 = query_engine_1.query(questions[0])
print("A:", response_1)
for node_with_score in response_1.source_nodes:
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
    print("-" * 30)

query_engine_5 = index.as_query_engine(similarity_top_k=5)
print("\nsimilarity_top_k=5")
response_5 = query_engine_5.query(questions[0])
print("A:", response_5)
for node_with_score in response_5.source_nodes:
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
    print("-" * 30)

# With similarity_top_k=1, the model used only the most relevant benefits
    # chunk and gave a concise, general answer. With similarity_top_k=5, the answer
    # became more detailed, but several unrelated chunks (security and financials were also retrieved. 
    # This shows that more retrieved context is
    # not always better because irrelevant information can be included and may
    # distract the model.

## Q3
question = "How is BrightLeaf doing overall as a company?"
query_engine = index.as_query_engine(similarity_top_k=5)
print(f"\nQ: {question}")
response = query_engine.query(question)
print("A:", response)
for node_with_score in response.source_nodes:
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node_with_score.node.get_content()}...")
    print("-" * 30)

# I decided to just ask it aobut the wellbeing of the company since i thought it would
    # need to explore various documents (earnings report, employee benefits). And to my surprised
    # it did a solid job of getting that information by describing the mission statement, financial performance
    # and the employee benefits. t also mentioned a parnership with another company which si a big deal
    # about renewable infrastructure which was cool to see. I decided to not limit the amount of tokens it would spew  
    # out which of course it mentioned some unneeded information like security policies. I think the best way to improve it
    # would be targeted retrieval and a different filtering/ranking system.

## Q4
llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)
q = "What employee benefits does BrightLeaf offer?"
response = query_engine.query(q)
faithfulness_result = faithfulness_evaluator.evaluate_response(
    query=q,
    response=response
)
relevancy_result = relevancy_evaluator.evaluate_response(
    query=q,
    response=response
)

print(" q1 ")
print("Question:", q)
print("Answer:", response)
print("Faithfulness Score:", faithfulness_result.score)
print("Relevancy Score:", relevancy_result.score)


q2 = "What is BrightLeaf's favorite sports team?"
response2 = query_engine.query(q2)
faithfulness_result2 = faithfulness_evaluator.evaluate_response(
    query=q2,
    response=response2
)
relevancy_result2 = relevancy_evaluator.evaluate_response(
    query=q2,
    response=response2
)
print("q2")
print("Question:", q2)
print("Answer:", response2)
print("Faithfulness Score:", faithfulness_result2.score)
print("Relevancy Score:", relevancy_result2.score)


# Analysis:
# A faithfulness score of 1.0 means the answer is fully supported by the
    # retrieved source context. A score of 0.0 indicates that the answer is not
    # supported by the retrieved context and may contain false/made up info.
# A relevancy score measures how well the response answers the user's question.
    # Does the answer actually address the question? 

# I think the scores should be higher for the employee-benefits question because
    # that information is clearly present in the BrightLeaf documents. The scores
    # may be lower for the sports-team question because that information is not in
    # the documents and is unrelated to the docuemnts.

# The first query scored 1.0 for both faithfulness and relevancy because the
    # answer was supported by the documents and directly answered the question.
# The second scored 0.0 for faithfulness because the documents had no answer,
    # but 1.0 for relevancy because the model correctly said the information was unavailable.
    # This makes since that information isn't in the documents and it cannot answer. 

# LLM-as-a-judge means using another language model to evaluate the quality of
    # an LLM's response against the question and retrieved context. It's useful since there are
    # many open-ended responees, not just one way to answer somethnig correctly so using 
    # LLM as a judge is a good way to measure instead of a classical accuacy metric which is more strict.
    