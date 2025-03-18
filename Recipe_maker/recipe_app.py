import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# --- 1. Load Hugging Face Model ---
generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")
embedding_model = SentenceTransformer('all-mpnet-base-v2')

# --- 2. Initialize ChromaDB ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="recipe_collection")

# --- 3. Load Recipe Dataset ---
dataset_path = "recipes.csv"
try:
    recipes_df = pd.read_csv(dataset_path)
except FileNotFoundError:
    st.error(f"Error: File not found at {dataset_path}")
    st.stop()

# Ensure necessary columns exist
required_columns = {"title", "ingredients", "img_src"}
if not required_columns.issubset(recipes_df.columns):
    st.error(f"Missing columns: {required_columns - set(recipes_df.columns)}")
    st.stop()

recipes_df.fillna("", inplace=True)
recipes_df["ingredients"] = recipes_df["ingredients"].str.lower().str.replace(r'[^\w\s]', '', regex=True)
recipes_df["combined_text"] = recipes_df["title"] + " " + recipes_df["ingredients"]

# --- 4. Store Recipe Embeddings in ChromaDB ---
def get_embeddings(text):
    """Generates embeddings using SentenceTransformer."""
    return embedding_model.encode(text).tolist()

existing_data = collection.get()
existing_ids = set(existing_data["ids"]) if existing_data and "ids" in existing_data else set()

for index, row in recipes_df.iterrows():
    recipe_id = str(index)
    if recipe_id in existing_ids:
        continue
    embedding = get_embeddings(row["combined_text"])
    collection.add(embeddings=[embedding], documents=[row["combined_text"]], ids=[recipe_id])

# --- 5. Retrieve Recipes from ChromaDB ---
def retrieve_recipes(query, top_k=5):
    """Retrieve top matching recipes from ChromaDB using embeddings."""
    query_embedding = get_embeddings(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    
    if results and "documents" in results:
        recipe_indices = [int(id) for id in results["ids"][0]]
        return recipes_df.iloc[recipe_indices]
    return None

# --- 6. Generate Recipe Using Hugging Face + RAG ---
def generate_recipe_rag(user_query, retrieved_recipes):
    """Generates a recipe using retrieved data (RAG approach)."""
    if retrieved_recipes is None or retrieved_recipes.empty:
        return "No relevant recipes found."

    # Extract key ingredients from retrieved recipes
    relevant_ingredients = set()
    for _, recipe in retrieved_recipes.iterrows():
        relevant_ingredients.update(recipe["ingredients"].split())

    # RAG-enhanced prompt
    prompt = (
        f"Using the following key ingredients: {', '.join(relevant_ingredients)},\n"
        f"generate a new recipe relevant to: {user_query}.\n"
        "Format:\n"
        "- Recipe Name\n"
        "- Ingredients\n"
        "- Steps\n"
    )

    response = generator(prompt, max_length=300, num_return_sequences=1)
    return response[0]["generated_text"]

# --- 7. Streamlit UI ---
st.title("🍽️ AI Recipe Generator (Hugging Face + RAG)")

user_query = st.text_input("Enter a dish name or ingredients:", "")

if st.button("Find Recipe"):
    if user_query:
        retrieved_recipes = retrieve_recipes(user_query)

        if retrieved_recipes is not None and not retrieved_recipes.empty:
            st.subheader("🍴 Found Recipes:")
            for _, recipe in retrieved_recipes.iterrows():
                st.markdown(f"### {recipe['title']}")
                st.write(f"**Ingredients:** {recipe['ingredients']}")
                if "img_src" in recipe and recipe["img_src"].strip():
                    st.image(recipe["img_src"], caption=recipe["title"], use_column_width=True)

            # Generate AI-based recipe
            generated_recipe = generate_recipe_rag(user_query, retrieved_recipes)
            st.subheader("📝 AI-Generated Recipe:")
            st.write(generated_recipe)
        else:
            st.warning("⚠️ No relevant recipes found.")
