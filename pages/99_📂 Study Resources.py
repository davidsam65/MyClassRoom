import streamlit as st
import base64


def download_file(filename, link_name=None):
    file_path = f"resources/{filename}"
    with open(file_path, "rb") as f:
        data = f.read()

    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">📄 {link_name}</a>'
    st.markdown(href, unsafe_allow_html=True)

    return

@st.cache_resource
def books_download():
    with st.expander("Books & Magazines", expanded=False):
        #download_file("Hands_On_ML_Aurelien_Geron.pdf", "Hand-on Machine Learning with Scikit-Learn, Keras, and TensorFlow")
        download_file("Prompt_eng_LLMs.pdf", "Prompt Engineering for LLMs")
        #download_file("Deep_Learning_by_Ian_Goodfellow.pdf", "Deep Learning by Ian Goodfellow")
        download_file("python_crash_course.pdf", "Python Crash Course")

    return


def main():
    st.markdown(
        """
        <h1 style='color: orange;'>Download Resources:</h1>
        """,
        unsafe_allow_html=True
    )

    books_download()

    with st.expander("SE Ai Item Search:", expanded=False):
        download_file("Ai_Search_structure.pdf", "Basic Structure of SE Ai Item Search")
        download_file("ER_Ai_Search.pdf", "ER Diagram of Ai Search")
        download_file("Cosine_Euclideon_Similarity_Theory.pdf", "Theory of Cosine and Euclidean Similarity")

    with st.expander("Streamlit Class:", expanded=False):
        download_file("quarterly_canada_population.csv", "Quarterly Canada Population Data")


if __name__ == "__main__":
    main()

