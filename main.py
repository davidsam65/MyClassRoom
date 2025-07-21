import os
import streamlit as st

st.set_page_config(
        page_title="AI Class Portal",
        page_icon="🏫",
        layout="wide"
    )


def main():
    st.image(os.path.join("assets", "ai_banner.png"), use_container_width=True)

    # st.title("👨‍🏫 Welcome to AI Engineering with Python")
    st.markdown("""
    This portal is built for students and developers learning **Python for AI Engineering**.
    """)

    return

if __name__ == "__main__":
    main()
