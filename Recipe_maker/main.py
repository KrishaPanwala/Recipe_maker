import streamlit as st
import pandas as pd
import chromadb
import torch
from transformers import AutoModel, AutoTokenizer
from diffusers import StableDiffusionPipeline

import google.generativeai as genai

# --- 1. Setup API Keys and Models ---
GEMINI_API_KEY = "AIzaSyCw5aL6gtJ4W0gWEfkT7GJ05VkUwDonqJo"  # Replace with your Gemini API key
genai.configure(api_key=GEMINI_API_KEY)

# Load Hugging Face embedding model
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
model = AutoModel.from_pretrained(embedding_model_name)

# Load Stable Diffusion for image generation (CPU-only)
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
st.write("CUDA is not available. Using CPU for image generation (this will be slow).")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="recipe_collection")

# --- 2. Load Dataset ---
dataset_path = "recipes.csv"  # Ensure the CSV file exists

try:
    recipes_df = pd.read_csv(dataset_path)
except FileNotFoundError:
    st.error(f"Error: File not found at {dataset_path}")
    st.stop()

# Clean dataset
recipes_df = recipes_df.fillna("")
if "ingredients" in recipes_df.columns:
    recipes_df["ingredients"] = recipes_df["ingredients"].str.lower().str.replace(r'[^\w\s]', '', regex=True)

# Combine title and ingredients for embeddings
if "title" in recipes_df.columns and "ingredients" in recipes_df.columns:
    recipes_df["combined_text"] = recipes_df["title"] + " " + recipes_df["ingredients"]
elif "title" in recipes_df.columns:
    recipes_df["combined_text"] = recipes_df["title"]
elif "ingredients" in recipes_df.columns:
    recipes_df["combined_text"] = recipes_df["ingredients"]
else:
    st.error("Dataset must contain at least a 'title' or 'ingredients' column.")
    st.stop()

# --- 3. Generate Embeddings and Store in ChromaDB ---
def get_sentence_transformer_embeddings(text):
    """Generate embeddings using Hugging Face transformers and return as a flat list."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract CLS token embeddings and convert to a flat list
    return outputs.last_hidden_state[:, 0, :].squeeze().tolist()

# Get existing IDs to avoid duplicates
existing_ids = set(collection.get()["ids"]) if collection.get()["ids"] else set()

for index, row in recipes_df.iterrows():
    recipe_id = str(index)

    if recipe_id in existing_ids:
        continue

    embedding = get_sentence_transformer_embeddings(row["combined_text"])

    if embedding:
        collection.add(
            embeddings=[embedding],
            documents=[row["combined_text"]],
            ids=[recipe_id]
        )

# --- 4. Retrieval Function ---
def retrieve_recipes(query, top_k=5):
    """Retrieves similar recipes from ChromaDB."""
    query_embedding = get_sentence_transformer_embeddings(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    if results and "documents" in results and results["documents"]:
        return results["documents"][0]
    return None

# --- 5. Generation Function (Gemini Pro) ---
def generate_recipe(prompt):
    """Generates a recipe using Gemini Pro."""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content(prompt)
        return response.text if response else "No response from API."
    except Exception as e:
        return f"Error generating recipe: {e}"

# --- 6. Image Generation Function ---
def generate_recipe_image(recipe_name):
    """Generates an image using Stable Diffusion."""
    try:
        prompt = f"A realistic image of {recipe_name} beautifully plated, professional food photography"
        with st.spinner("Generating image..."):
            image = pipe(prompt, height=256, width=256, num_inference_steps=25).images[0]
            st.image(image, caption=f"Generated Image of {recipe_name}", use_column_width=True)
    except Exception as e:
        st.error(f"Error generating image: {e}")
        st.write(f"Error details: {str(e)}")

# --- 7. Streamlit UI ---
st.title("🍽️ AI Recipe Generator (Hugging Face & Gemini)")

user_query = st.text_input("Enter a dish name or ingredients:", "")

if st.button("Find Recipe"):
    if user_query:
        retrieved_recipes = retrieve_recipes(user_query)

        if retrieved_recipes:
            retrieved_text = "\n".join(retrieved_recipes)
            augmented_prompt = f"Generate a recipe for: {user_query} based on these recipes: \n{retrieved_text}"
            generated_recipe = generate_recipe(augmented_prompt)

            st.subheader("📝 Generated Recipe:")
            st.write(generated_recipe)

            generate_recipe_image(user_query)
        else:
            st.warning("⚠️ No relevant recipes found.")
