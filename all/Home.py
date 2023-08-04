import streamlit as st

# # Initialize session state variables
# if 'openai_api_key' not in st.session_state:
# 	st.session_state.openai_api_key = ""

# if 'serper_api_key' not in st.session_state:
# 	st.session_state.serper_api_key = ""

st.set_page_config(page_title="Home", page_icon="🦜️🔗")

st.header("Добро Пожаловать в TeachAI! 👋")

st.markdown(
    """
    Платформа TeachAI поможет вам планировать свое преподавание на занятиях, создавать сценарии и редактировать их, просто выбирая то, что вы хотите, платформа, которая настроит учебный план в соответствии с вашим стилем преподавания и академической успеваемостью учащихся🔥!
    
    ##### PDF Plan
    Загружайте свои текстовые материалы в PDF формате и получите результат в течение 2-5 минут в зависимости размера материалов


    #####  Youtube Plan
    Вводите ссылки ютую видео либо напишите тему в поле для того чтоб использовать транскрипт данных видео чтобы преобразовать его в учебный сценарий
    """
)

