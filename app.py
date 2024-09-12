import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer, util

# Load the tokenizer, model, and the SentenceTransformer model for similarity matching
@st.cache_resource
def load_models():
    # Load the fine-tuned PyTorch model
    model = GPT2LMHeadModel.from_pretrained('fine_tuned_headline_model')
    
    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('fine_tuned_headline_model')
    tokenizer.pad_token = tokenizer.eos_token  # To avoid padding issues
    
    # Load the sentence transformer for embedding comparison
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model for computing embeddings
    
    return tokenizer, model, embedding_model

# Predefined categories
categories = ['entertainment', 'business', 'sport', 'politics', 'tech']

# Function to compute category similarity
def compute_category_similarity(generated_text, categories, embedding_model):
    # Get embeddings for the generated text and each category
    generated_embedding = embedding_model.encode(generated_text, convert_to_tensor=True)
    category_embeddings = embedding_model.encode(categories, convert_to_tensor=True)

    # Compute cosine similarities between the generated text and the category names
    similarities = util.pytorch_cos_sim(generated_embedding, category_embeddings)
    
    # Get the category with the highest similarity score
    best_match_idx = similarities.argmax().item()
    return categories[best_match_idx]

# Streamlit App UI
st.title("Headline Categorization")
st.write("Input a news headline, and the app will classify it into a category.")

# Load models
tokenizer, model, embedding_model = load_models()

# Input: News Headline
headline_data = st.text_area("Enter a news headline")

if st.button("Categorize Headline"):
    if headline_data:
        # Preprocess the input text
        input_text = f"Headline: {headline_data} Category:"
        
        # Tokenize input
        input_ids = tokenizer.encode(input_text, return_tensors='pt')

        # Generate text from the model with a larger max_length or using max_new_tokens
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=50, num_return_sequences=1, no_repeat_ngram_size=2)

        # Decode the generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Use embedding comparison to predict the category
        predicted_category = compute_category_similarity(generated_text, categories, embedding_model)

        # Display the result
        st.write("### Predicted Category:")
        st.write(predicted_category.capitalize())
    else:
        st.write("Please enter a news headline.")
