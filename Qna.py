import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import BartTokenizer, BartForConditionalGeneration
from pinecone import Index
from sentence_transformers import SentenceTransformer
import pinecone


model_directory = "retrever"  # Specify the directory where you saved the model
retriever = SentenceTransformer(model_directory)

# Initialize Pinecone index
api_key = "api_key"
pinecone.init(api_key=api_key, environment="gcp-starter")
index_name = "question-answer-app"
index = Index(index_name)

# Initialize BART models
tokenizer_directory = "bart_tokenizer"
generator_directory = "bart_generator"

# Load the models from the specified directories
tokenizer = BartTokenizer.from_pretrained(tokenizer_directory)
generator = BartForConditionalGeneration.from_pretrained(generator_directory)

truncation_strategy = 'longest_first'

@st.cache
def query_pinecone(query, top_k):
    q = retriever.encode([query]).tolist()
    context = index.query(q, top_k=top_k, include_metadata=True)
    return context

@st.cache
def format_query(query, context):
    context = [f"<P> {m['metadata']['passage_text']}" for m in context]
    context = " ".join(context)
    query = f"question: {query} context: {context}"
    return query

@st.cache
def generate_answers(query):
    inputs = tokenizer([query], max_length=1024, return_tensors="pt",truncation=truncation_strategy)
    ids = generator.generate(inputs["input_ids"], num_beams=2, min_length=20, max_length=60)
    answer = tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return answer

st.title("History Question Answering")

# Input for the user's query
user_query = st.text_input("Enter your question:")
st.write("Eg:- What is Taj Mahal")

# Button to trigger the query and answer generation
if st.button("Get Answer"):
    st.write("Searching for answers...")
    user_query = user_query.lower()
    q_result = query_pinecone(user_query, top_k=5)
    context = format_query(user_query, q_result['matches'])
    answer = generate_answers(context)

    st.write("Answer:", answer)

st.write("Note: Model only have data before 2019. Questions after that year may result in wrong data")
