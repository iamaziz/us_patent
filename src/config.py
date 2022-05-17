import streamlit as st


def APP_PAGE_HEADER():
    st.set_page_config(
        page_title="U.S. Patent", page_icon="🔬", layout="wide", initial_sidebar_state="collapsed"
    )

    hide_streamlit_style = """
                    <style>
                    #MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    </style>
                    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
