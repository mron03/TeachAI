import streamlit as st

# # Initialize session state variables
# if 'openai_api_key' not in st.session_state:
# 	st.session_state.openai_api_key = ""

# if 'serper_api_key' not in st.session_state:
# 	st.session_state.serper_api_key = ""

st.set_page_config(page_title="Home", page_icon="🦜️🔗")

st.header("Welcome to TeachingAI! 👋")

st.markdown(
    """
    TeachAI platform will help you to plan your teaching in classes, build scenarios and edit them just by saing what you want, platform that will configure teaching plan according to your teaching style and student academic performance🔥!
    
    ##### Build your scenario based on your PDF and Youtube Videos
    """
)
