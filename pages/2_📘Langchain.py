import streamlit as st

st.set_page_config(
        page_title="Langchain",  # Page title shown in the browser tab
        page_icon="🏠",              # Icon shown in the browser tab
        layout="wide",
        initial_sidebar_state="expanded",
    )

st.markdown("<h3 style='color: orange;'>Langchain 01</h3>", unsafe_allow_html=True)

with st.expander("Introduction to Langchain", expanded=False):
    code_lc_basic = """
    import streamlit as st
    
    from langchain_ollama import ChatOllama
    from langchain_openai import ChatOpenAI
    
    import os
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    
    @st.cache_resource
    def load_llm_model(provider):
        if provider == "Ollama":
            llm = ChatOllama(
                model="llama3.2",
                temperature=0.5
            )
        else:
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.5,
                max_tokens=2000,
            )
    
        return llm
    
    def get_completion(prompt, llm):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    
        response = llm.invoke(messages)
    
        return response.content
    
    
    def select_expert()->str:
        experts = [
            "General",
            "Mathematics",
            "Finance",
            "Information Technology",
            "Food and Nutrition",
            "Programming",
            "Business",
        ]
    
        expert = st.sidebar.selectbox("Select an Expert", experts, key="expert")
    
        return expert
    
    def get_completion_expert(expert, prompt, llm):
        system_prompt =f"You are a helpful assistant with expertise in {expert}."
    
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    
        response = llm.invoke(messages)
    
        return response.content
    
    def main():
        st.set_page_config(
            page_title="LC - 01",  # Page title shown in the browser tab
            page_icon="🏠",  # Icon shown in the browser tab
            layout="wide",
            initial_sidebar_state="expanded",
        )
    
        st.markdown("<h1 style='color: orange;'>LangChain and LLM - Deep Dive</h1>", unsafe_allow_html=True)
        #-- Langchain Language Model
        provider = st.sidebar.selectbox("Select Provider", ["Ollama", "OpenAi"], key="provider")
        llm = load_llm_model(provider)
        selected_expert = select_expert()
    
        ai_tone = st.sidebar.selectbox("Tone", ["Polite", "Casual", "Formal"], key="tone")
        ai_lang = st.sidebar.selectbox("Language", ["Thai", "Chinese", "French", "Vietnam"], key="lang")
    
        customer_review = "
        Your product is terrible!. I don't know how you were able to sell this product. I don't want to use it again! Actually, no one should want this.
        Seriously, Give me my money back now!.
        "
        st.text_area("Review", customer_review, height=140)
    
        prompt_text = f"
        Rewrite the following: \\n{customer_review}\\n
        in a '{ai_tone}' tone, and then please translate the new review message into '{ai_lang}' language.
        "
    
        st.text_area("Prompt:", prompt_text, height=240)
    
    
        if st.button("Ai"):
            complete = get_completion(prompt_text, llm)
            # rewrite = get_completion_expert(selected_expert, prompt_text, llm)
            st.text(complete)
    
        return
    """
    st.code(code_lc_basic, language="python")

with st.expander("Langchain Template", expanded=False):
    code_lc_template = """
    import streamlit as st
    
    from langchain_ollama import ChatOllama
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    
    import os
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    
    chat = ChatOllama(
        model="llama3.2",
        temperature=0.5
    )
    
    def ini_page():
        st.set_page_config(
                page_title="LC - 02",  # Page title shown in the browser tab
                page_icon="🏠",  # Icon shown in the browser tab
                layout="wide",
                initial_sidebar_state="auto",
            )
    
        st.markdown("<h1 style='color: orange;'>LangChain and LLM - Template</h1>", unsafe_allow_html=True)
        return
    
    def lc_template():
        template = "
                As a children's book author, you are tasked with writing a story that is engaging and educational for young readers.
                Please come up with a simple and short (90 words) lullaby based on the location {location} and the main character {name}.  
    
                STORY: 
            "
    
        prompt = PromptTemplate(
            input_variables=[
                "location",
                "name"
            ],
            template=template
        )
    
        llm_chain = LLMChain(llm=chat, prompt=prompt)
        return llm_chain
    
    def main():
        ini_page()
    
        story_chain = lc_template()
    
        template_variables = {
            "location": "Zanzibar",
            "name": "MAYA"
        }
    
        result = story_chain.invoke(template_variables)
        st.write(result["text"])
    
        return

    """

    st.code(code_lc_template, language="python")
