import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
        page_title="Courses",  # Page title shown in the browser tab
        page_icon="🏠",              # Icon shown in the browser tab
        layout="wide",
        initial_sidebar_state="expanded",
    )


st.markdown("<h3 style='color: orange;'>Siam East Item Code Ai Search DEMO:</h3>", unsafe_allow_html=True)

def main():
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
    
        """
        st.code(code_embedding)

    with st.expander("- Similarity Search"):
        st.markdown("<h3 style='color: green;'> Cosine Similarity Between Two Vectors A and B </h3>", unsafe_allow_html=True)

        st.latex(r"""
        \text{Cosine Similarity} = \cos(\theta) = 
        \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \times \|\mathbf{B}\|}
        """)

        st.markdown("**Where:**")

        st.latex(r"""
        \mathbf{A} \cdot \mathbf{B} = \sum_{i=1}^{n} A_i \cdot B_i \quad \text{(Dot product)}
        """)

        st.latex(r"""
        \|\mathbf{A}\| = \sqrt{\sum_{i=1}^{n} A_i^2} \quad \text{(Magnitude of vector A)}
        """)

        st.latex(r"""
        \|\mathbf{B}\| = \sqrt{\sum_{i=1}^{n} B_i^2} \quad \text{(Magnitude of vector B)}
        """)

        code_similarity = """ Python Code as follows:
        
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

    with st.expander("- Euclidean Distance Between Two Vectors A and B (L2 Norm)"):
        st.markdown("<h3 style='color: green;'> Euclidean Distance Between Two Vectors A and B </h3>", unsafe_allow_html=True)

        st.latex(r"""
        \text{Euclidean Distance} = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2}
        """)

        st.markdown("**Where:**")

        st.latex(r"""
        A_i \text{ and } B_i \text{ are the components of vectors A and B respectively.}
        """)

        code_euclidean = """ Python Code as follows:
        
        def euclidean_distance(a, b):
            return np.sqrt(np.sum((a - b) ** 2))
        
        """
        st.code(code_euclidean)


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

    st.markdown("<h3 style='color: orange;'>Cosine Similarity and Euclidean Distance Example:</h3>",
                unsafe_allow_html=True)

    with st.expander("- Cosine Similarity and Euclidean Distance Example Python Code"):
        ex_code = r"""
        import numpy as np

        A = np.array([1, 2, 3])
        B = np.array([4, 5, 6])

        cos_sim = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
        euclid_dist = np.linalg.norm(A - B)

        st.markdown("### 🔢 Vectors:")
        st.latex(r"\mathbf{A} = [1, 2, 3], \quad \mathbf{B} = [4, 5, 6]")

        st.markdown("### 📐 Cosine Similarity Result:")
        st.latex(f"\\cos(\\theta) \\approx {cos_sim:.4f}")

        st.markdown("### 📏 Euclidean Distance Result:")
        st.latex(f"d(\\mathbf{{A}}, \\mathbf{{B}}) = {euclid_dist:.4f}")

        """
        st.code(ex_code, language='python')

        if st.button("Run-Plot Cosine and Euclidean Example"):
            example_cosine_euclidean()
            example_cosine_euclidean_plot()

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


        return

def example_cosine_euclidean():
    A = np.array([1, 2, 3])
    B = np.array([4, 5, 6])

    cos_sim = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    euclid_dist = np.linalg.norm(A - B)

    st.markdown("### 🔢 Vectors:")
    st.latex(r"\mathbf{A} = [1, 2, 3], \quad \mathbf{B} = [4, 5, 6]")

    st.markdown("### 📐 Cosine Similarity Result:")
    st.latex(f"\\cos(\\theta) \\approx {cos_sim:.4f}")

    st.markdown("### 📏 Euclidean Distance Result:")
    st.latex(f"d(\\mathbf{{A}}, \\mathbf{{B}}) = {euclid_dist:.4f}")

    return cos_sim, euclid_dist

def example_cosine_euclidean_plot():
    A = np.array([1, 2, 3])
    B = np.array([4, 5, 6])

    # Calculate cosine similarity and Euclidean distance
    cos_sim = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    euclid_dist = np.linalg.norm(A - B)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Project 3D vectors to 2D (X, Y only for visualization)
    A_2d = A[:2]
    B_2d = B[:2]

    # Plot vectors
    ax.quiver(0, 0, A_2d[0], A_2d[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Vector A')
    ax.quiver(0, 0, B_2d[0], B_2d[1], angles='xy', scale_units='xy', scale=1, color='red', label='Vector B')

    # Plot dashed line (Euclidean distance) between tips of A and B
    ax.plot([A_2d[0], B_2d[0]], [A_2d[1], B_2d[1]], 'k--', label='Euclidean Distance')

    # Set limits and labels
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 6)
    ax.set_xlabel('X-axis', fontsize=6)
    ax.set_ylabel('Y-axis', fontsize=6)
    ax.axhline(0, color='black', linewidth=0.5, ls='--')
    ax.axvline(0, color='black', linewidth=0.5, ls='--')

    # Add grid, legend, and title with computed values
    ax.grid(True)

    # Ticks font size
    ax.tick_params(axis='both', which='major', labelsize=6)

    ax.legend(fontsize=6)
    ax.set_title(f'Cosine Similarity ≈ {cos_sim:.4f} | Euclidean Distance ≈ {euclid_dist:.4f}', fontsize=6)

    # Display the plot in Streamlit
    st.pyplot(fig)


if __name__ == "__main__":
    main()



