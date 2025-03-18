import streamlit as st
import google.generativeai as genai
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import requests
from PIL import Image
from io import BytesIO

# --- 1. Setup Gemini API and ChromaDB ---
GEMINI_API_KEY = "AIzaSyBFx-3LRQRPFE8t2dTzQe368SQEvlHcDck"  # Replace with your Gemini API key 
genai.configure(api_key=GEMINI_API_KEY)

# Initialize embedding model
embedding_model = SentenceTransformer('all-mpnet-base-v2')

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

# Fix column names (rename 'recipe_name' to 'title' for consistency)
if "recipe_name" in recipes_df.columns:
    recipes_df.rename(columns={"recipe_name": "title"}, inplace=True)

# Ensure dataset contains required columns
required_columns = {"title", "ingredients", "img_src"}
missing_columns = required_columns - set(recipes_df.columns)

if missing_columns:
    st.error(f"Missing columns in dataset: {missing_columns}")
    st.stop()

# Clean dataset
recipes_df = recipes_df.fillna("")
recipes_df["ingredients"] = recipes_df["ingredients"].str.lower().str.replace(r'[^\w\s]', '', regex=True)

# Combine title and ingredients for embeddings
recipes_df["combined_text"] = recipes_df["title"] + " " + recipes_df["ingredients"]

# --- 3. Generate Embeddings and Store in ChromaDB ---
def get_sentence_transformer_embeddings(text):
    """Generates embeddings using SentenceTransformer."""
    return embedding_model.encode(text).tolist()

# Ensure ChromaDB collection exists
try:
    existing_data = collection.get()
    existing_ids = set(existing_data["ids"]) if existing_data and "ids" in existing_data else set()
except Exception as e:
    st.error(f"⚠️ Error fetching ChromaDB data: {e}")
    existing_ids = set()

# Add new embeddings only if not already in ChromaDB
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

# --- 4. Retrieval Function (Improved with Filtering) ---
def retrieve_recipes(query, top_k=5):
    """Retrieves most relevant recipes based on ingredient match."""
    query_embedding = get_sentence_transformer_embeddings(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    if results and "documents" in results and results["documents"]:
        recipe_indices = [int(id) for id in results["ids"][0]]

        # Filter by cosine similarity threshold (e.g., 0.8)
        filtered_recipes = []
        for index in recipe_indices:
            score = results["distances"][0][recipe_indices.index(index)]
            if score >= 0.8:  # Only include highly relevant matches
                filtered_recipes.append(recipes_df.iloc[index])

        return pd.DataFrame(filtered_recipes) if filtered_recipes else None
    return None

# --- 5. Generation Function (Refined Prompt) ---
def generate_recipe(user_query, retrieved_recipes):
    """Generates a recipe using Gemini Pro with a structured prompt."""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')

        # Extract key details from retrieved recipes
        relevant_ingredients = set()
        for _, recipe in retrieved_recipes.iterrows():
            relevant_ingredients.update(recipe["ingredients"].split())

        # Structured prompt
        structured_prompt = (
            f"Create a new recipe using these key ingredients: {', '.join(relevant_ingredients)}.\n"
            f"Ensure the recipe is relevant to: {user_query}.\n"
            "Format: \n"
            "- Recipe Name\n"
            "- Ingredients\n"
            "- Steps\n"
        )

        response = model.generate_content(structured_prompt)
        return response.text if response else "No response from API."
    except Exception as e:
        return f"Error generating recipe: {e}"

# --- 6. Image Display Function ---
def display_image(image_url, recipe_name):
    """Fetches and displays an image from a URL."""
    try:
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        image = Image.open(BytesIO(image_response.content))
        st.image(image, caption=recipe_name, use_container_width=True)
    except requests.exceptions.RequestException:
        st.warning(f"⚠️ Could not fetch image for {recipe_name}")

# --- 7. Streamlit UI ---
st.title("🍽️ AI Recipe Generator")

# User Input
user_query = st.text_input("Enter a dish name or ingredients:", "")

if st.button("Find Recipe"):
    if user_query:
        retrieved_recipes = retrieve_recipes(user_query)

        if retrieved_recipes is not None and not retrieved_recipes.empty:
            st.subheader("🍴 Found Recipes:")
            for _, recipe in retrieved_recipes.iterrows():
                st.markdown(f"### {recipe['title']}")
                st.write(f"**Ingredients:** {recipe['ingredients']}")
                
                # Display image if available
                if "img_src" in recipe and recipe["img_src"].strip():
                    display_image(recipe["img_src"], recipe["title"])
                else:
                    st.warning("⚠️ No image available")

            # Generate AI-based recipe
            generated_recipe = generate_recipe(user_query, retrieved_recipes)

            st.subheader("📝 AI-Generated Recipe:")
            st.write(generated_recipe)

        else:
            st.warning("⚠️ No relevant recipes found.")
