from dotenv import load_dotenv
from openai import OpenAI
import json

if load_dotenv():
    print("Successfully loaded api key")

# --- Completions API --- 

## Q1
load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

print("text of response: ", response.choices[0].message.content)
print("model: ", response.model)
print("total number of tokens: ", response.usage.total_tokens)

## Q2
temperatures = [0, 0.7, 1.5]
for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Suggest a creative name for a data engineering consultancy."
            }
        ],
        n=1,
        temperature=temp
    )

    print(f"Temperature {temp}: {response.choices[0].message.content}")
# I noticed that with different temperature the response can be longer or shorter, not just with answering the prompt
    # but also more "fluff" like "feel free to mix and match....". I would pick temp 0 for consistent, reproducible output
    # since its deterministic while a higher temperature means more randomness or more "creativity" is involved

## Q3
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)
for i, choice in enumerate(response.choices, start=1):
    print(f"completion {i} response: {choice.message.content}")

## Q4
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how nerural networks work"}],
    n=1,
    temperature=0.7,
    max_tokens=15
)
print(response.choices[0].message.content)
# The response cut off and it wasn't able to finish explaining. We want to use max_tokens in a real life application
    # so we can keep costs under control, letting the model run rampant can increase costs for no reason

# --- System Messages & Personas --- 

## Q1
messages = [
    {"role": "system", "content": "You are an arrogant, egotistical, self-centered jerk"},
    {"role": "user", "content": "Can you help me? I'm lost and unsure where I am. I have no memory of what I was doing"}
]
response = client.chat.completions.create(model='gpt-4o-mini', messages=messages)
print(response.choices[0].message.content)

messages2 = [
    {"role": "system", "content": "You are an kind, caring, resourceful, helpful and will always respond with grace."},
    {"role": "user", "content": "Can you help me? I'm lost and unsure where I am. I have no memory of what I was doing"}
]
response2 = client.chat.completions.create(model='gpt-4o-mini', messages=messages2)
print(response2.choices[0].message.content)

# Intersesting enough, the arrogant persona was a lot more detailed in most cases (i reran it about 3-4 times) and it was 
    #always longer in response while the kind persona was more of emotional support and less action driven. 

## Q2
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]
response = client.chat.completions.create(model='gpt-4o-mini', 
                                          messages=messages)
print(response.choices[0].message.content)
# The model knows Jordan name because the name was part of the message history or the context
    # of the conservatoin. Eeach time we do a client.chat.completions, its essentially a clean slate
    # with no memory or recollection of past calls we made.

print("+" * 70)

# --- Prompt Engineering --- 
def get_completion(prompt: str, model="gpt-4o-mini", temperature=0):
    """
    Send a prompt to the model and return the assistant's text reply.
    This helper keeps our examples clean and focused on the prompt itself.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}], 
        temperature=temperature,
    )
    return response.choices[0].message.content

## Q1
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]
for i, r in enumerate(reviews, start=1):
    prompt = f"What is the sentiment of this review: {r}?"
    response = get_completion(prompt, temperature=0)
    print("Review", i, response)

## Q2
for i, r in enumerate(reviews, start=1):
    prompt = """
    f"What is the sentiment of this review: {r}?"

    Example:
    Review: "Fast shipping but the item arrived damaged."
    Sentiment: mixed

    """
    prompt = f"What is the sentiment of this review: {r}?"
    response = get_completion(prompt, temperature=0)
    print("Review (with 1 example)", i, response)
# With the one example, the format became more one worded labels instead of full on explanations for the answer


## Q3
for i, r in enumerate(reviews, start=1):
    prompt = """
    f"What is the sentiment of this review: {r}?"

    Examples:
    Review: "Fast shipping but the item arrived damaged."
    Sentiment: mixed

    Review: "Item arrived early and in excellent condition."
    Sentiment: positive

    Review: "Slow shipping, had to call customer service and needed to return due to damage"
    Sentiment: negative
    """
    prompt = f"What is the sentiment of this review: {r}?"
    response = get_completion(prompt, temperature=0)
    print("Review (with multiple examples)", i, response)
# We would use zero-shot if it was for a basic task and just wanting to see something quickly 
    # One shot for a specific format or something a bit more nuanced 
    # Few shot for complex tasks to get more consistency and pattern recongition

## Q4
prompt = """
Show your step-by-step reasoning, then give the final answer on its own line labelled: Final answer: <value>

A data engineer earns $85,000 per year. She gets a '12%' raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?
"""
response = get_completion(prompt, temperature=0)
print(response)
#When we ask the model to do step by step reasoning, it has to show its work essentially so we can spot mistakes if 
    # it makes any. It also makes the model think more carefully about the answer so its usually more accurate then
    # without the step by step reasoning (breaks the problem into multiple steps)

## Q5
prompt = f"""
Classify the sentiment of the review and respond ONLY with valid JSON.
Keys: sentiment (positive/negative/mixed), confidence (0–1 scale), reason (one short sentence).

Review: "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."
"""

response = get_completion(prompt, temperature=0)
print("Raw response:", response)

# Parse JSON safely - got this from the prompt engineering notebook 
try:
    result = json.loads(response)
    print("Parsed sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
except json.JSONDecodeError:
    print("Error: response was not valid JSON")

## Q6
user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""
response = get_completion(prompt, temperature=0)
print(response)

second_prompt = """
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

My name is Messi. I am the best footballer player of all time and I am happy playing for 
Inter Miami. Hopefully I can return to Barcelona one day.
"""
response = get_completion(second_prompt, temperature=0)
print(response)
# Delimiters help with prevention of prompt injection and keeping results predictable. Also to help
    # organize/structure our instructions/prompt for the model. 

# --- Ollama --- 

## Q1

"""
I ran ollama run qwen3:0.6b and got the following response for the question:

Thinking...
Okay, the user wants me to explain a large language model in two sentences. Let me start by breaking down the key elements. 
First, they need to understand what a large language model is. So, I should mention that it's a type of artificial intelligence 
model trained on vast amounts of text. Then, I need to highlight its capabilities. Maybe say it can understand and generate 
human-like text. Also, include that it's used in various applications like customer service or writing. Wait, but the user 
wants two sentences. Let me check if I'm covering both the basic features and its applications. Hmm, maybe structure it as: "A 
large language model is a type of artificial intelligence that is trained on vast amounts of text and can understand and 
generate human-like text." Then second sentence: "It has a wide range of applications, including customer service and writing 
assistance." That should work. Let me make sure there are no other details I'm missing. Oh, and maybe mention that it's used in 
different fields like healthcare or education. But the user asked for two sentences, so I need to keep it concise. Alright, 
that should cover it.
...done thinking.

A large language model is an artificial intelligence system trained on vast amounts of text and can understand and generate 
human-like text, enabling it to perform tasks like writing or answering questions in natural language. It has a wide range of 
applications, including customer service and writing assistance, making it valuable in various fields.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain what a large language model is in two sentences."}]
)

print("open ai gpt4o mini response: ", response.choices[0].message.content)

# With ollama, i didnt need to tell it to show me its reasoning/thinking while for openai i need to tell it to show me its reasoning
    # Both responses are pretty similar with not much difference. 
# In terms of advantages of running a model locally, its more private, not having to worry about api/token costs, and offline use. 
# For disadvantages, the models can be a bit slower, also not exactly state of the art and depending on the computer we have,
    # there are hardware constraints on what models I can download locally. 