import streamlit as st


def APP_PAGE_HEADER():
    st.set_page_config(
        page_title="U.S. Patent", page_icon="🔬", layout="wide", initial_sidebar_state="collapsed"
    )
    st.markdown(
        "### [U.S. Patent Phrase to Phrase Matching](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching)",
        unsafe_allow_html=True)
    hide_streamlit_style = """
                    <style>
                    #MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    </style>
                    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
