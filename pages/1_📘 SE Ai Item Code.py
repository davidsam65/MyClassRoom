import streamlit as st

st.set_page_config(
        page_title="Courses",  # Page title shown in the browser tab
        page_icon="🏠",              # Icon shown in the browser tab
        layout="wide",
        initial_sidebar_state="expanded",
    )

# st.markdown("<h1 style='color: orange;'>📘 Course Curriculums</h1>", unsafe_allow_html=True)


# ### Week 3: LLM & Langchain Class#1
# - Neural Networks with TensorFlow/Keras
# - CNN, RNN, LSTM, Transformers
#
# ### Week 4: Real-world Applications
# - Chatbots (LangChain, RAG)
# - AI for Business Automation
# """)

st.markdown("<h3 style='color: orange;'>Siam East Item Code Ai Search DEMO:</h3>", unsafe_allow_html=True)

with st.expander("- Embedding, Vector Databases"):
    code_embedding = """
    import streamlit as st
    import embeddings as embeddings
    from sentence_transformers import SentenceTransformer
    import os
    
    # Cached resource loader — accepts model_name as input
    @st.cache_resource
    def load_model(name):
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    
        if name == "Fine-Tuned":
            path = os.path.join(base_path, "fine_tuned_item_embedder")
        else:
            path = os.path.join(base_path, name)
    
        if os.path.exists(path):
            st.info(f"📦 Loading model from: `{path}`")
            return SentenceTransformer(path)
        else:
            st.warning(f"⚠️ Model `{name}` not found locally. Loading fallback: `LaBSE`")
            return SentenceTransformer(os.path.join(base_path, "LaBSE"))
            
    def main():
    st.set_page_config(
        page_title="Embedding",  # Page title shown in the browser tab
        page_icon="🏠",  # Icon shown in the browser tab
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("<h1 style='color: orange;'>Embedding Vector</h1>", unsafe_allow_html=True)

    # 1️⃣ UI outside the cached function
    model_name = st.selectbox("Select embedding model:", [
        # "BAAI/bge-m3" #1024,
        # "distiluse-base-multilingual-cased-v1" #512,
        "LaBSE",  # 768
        "Fine-Tuned"
    ])

    # 3️⃣ Load the model
    model = load_model(model_name)

    if st.button("Embedding"):
        progress = st.progress(0, text="Loading data from Postgres...")

        # Step 1: Load data
        df = embeddings.load_data_from_postgres()
        progress.progress(20, text="Data loaded. Computing embeddings...")


        # Step 2: Compute embeddings
        df = embeddings.compute_embeddings(df, model)
        progress.progress(70, text="Embeddings computed. Saving model...")

        # Step 3: Store embeddings to Postgres
        embeddings.store_embeddings_to_postgres(df)
        progress.progress(100, text="Embeddings stored in database successfully ✅")

        st.success("All steps completed.")

    return

    if __name__ == "__main__":
        main()

    """
    st.code(code_embedding)

with st.expander("- Similarity Search"):
    code_similarity = """
    def search_similar_items(user_query, model, top_k=5):
    # Convert vector to native Python list ✅
    query_vector = model.encode(user_query).tolist()

    query_embedding_str = "'[" + ", ".join(map(str, query_vector)) + "]'"

    sql = f"
        select e.code, m.express_desc, m.full_desc, e.embedding <=> {query_embedding_str}::vector AS similarity_score
        from item_embeddings e
        join item_master m on e.code = m.code
        order by similarity_score ASC
        limit {top_k};
    "

    db = util_db.PostgresDB()
    db.connect()
    df = db.fetch_dataframe(sql)
    db.disconnect()

    return df
    """
    st.code(code_similarity)

with st.expander("- Feedback for Fine-Tuning"):
    code_feedback = """
        def insert_feedback(self, query_text, selected_code):
            sql = "
                  INSERT INTO feedback_log (query, selected_code, timestamp)
                  VALUES (%s, %s, %s) \
                  "
            params = (query_text, selected_code, datetime.now())
            try:
                with self.connection.cursor() as cur:
                    cur.execute(sql, params)
                    self.connection.commit()
                    logger.info(f"Inserted feedback: {selected_code}")
            except Exception as e:
                self.connection.rollback()
                logger.error(f"Insert feedback failed: {e}")
                raise
                
        def load_feedback_with_items() -> pd.DataFrame:
            db = util_db.PostgresDB()
            db.connect()
        
            query = "
                SELECT f.query, i.code, i.unit, i.type, i.express_desc, i.full_desc
                FROM feedback_log f
                JOIN item_master i ON f.selected_code = i.code
            "
            df = db.fetch_dataframe(query)
            db.disconnect()
        
            df = df.fillna('')
            df['positive'] = df[['code', 'unit', 'type', 'express_desc', 'full_desc']].agg(' '.join, axis=1)
            return df[['query', 'positive']]
    
        def build_training_examples(df: pd.DataFrame) -> list:
            return [
                InputExample(texts=[row['query'], row['positive']], label=1.0)
                for _, row in df.iterrows()
            ]
    
        def fine_tune_model(
            model_name: str = 'sentence-transformers/LaBSE',
            output_dir: str = 'models/fine_tuned_item_embedder',
            batch_size: int = 8,
            epochs: int = 1
        ) -> str:
            
            # Load and prepare data
            df = load_feedback_with_items()
            if df.empty:
                return "⚠️ No feedback data found. Aborting fine-tuning."
        
            train_examples = build_training_examples(df)
        
            # Load base model
            model = SentenceTransformer(model_name)
        
            # Use DataLoader with InputExample directly (modern usage)
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
            # Define loss function
            train_loss = losses.CosineSimilarityLoss(model)
        
            # Train the model
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=epochs,
                warmup_steps=10,
                show_progress_bar=True
            )
        
            # Save model
            os.makedirs(output_dir, exist_ok=True)
            model.save(output_dir)
        
            return f"Fine-tuned model saved to: {output_dir}"
    """
    st.code(code_feedback)


st.markdown("<h3 style='color: orange;'>LLM DEMO for Siam East Item Code Ai Search:</h3>", unsafe_allow_html=True)

with st.expander("- LLM Integration with Result return from Similarity Search"):
    code_model_loading = """
        import ollama
        from openai import OpenAI
        
        def query_sql(user_query, model, top_k=5, return_all=False):
    
            query_vector = model.encode(user_query).tolist()
        
            query_embedding_str = "'[" + ", ".join(map(str, query_vector)) + "]'"
        
            # Build SQL with optional LIMIT
            limit_clause = "" if return_all else f"LIMIT {top_k}"
        
            sql = f"
                    select context, embedding <=> {query_embedding_str}::vector AS similarity_score
                    from item_embeddings
                    order by similarity_score ASC
                    {limit_clause};
                "
        
            db = util_db.PostgresDB()
            db.connect()
            df = db.fetch_dataframe(sql)
            db.disconnect()
        
            return df
            
        def generate_llm_response(query_text, df, system_prompt, llm_model):
            context = "\\n"".join(df["context"].tolist())
        
            prompt = f"Answer the following question based on the context provided:\\n\\n{context}\\n\\nQuestion: {query_text}"
        
            if llm_model == "OpenAI":
                client = OpenAI(api_key=openai_api_key)
        
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=1,
                    top_p=1,
                    max_tokens=2000,
                    stream=False,
                    seed=100,
                )
        
                return response.choices[0].message.content  # ✅ Correct way
            else:
                response = ollama.chat(model="llama3.2", messages=[
                    {"role": "system", "content": "You are an expert in item code identification."},
                    {"role": "user", "content": prompt}
                ])
        
                return response["message"]["content"]
        """
    st.code(code_model_loading)

