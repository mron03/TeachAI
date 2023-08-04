import os, psycopg2
import requests

import streamlit as st

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO




def generate_pdf(text):
    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    # Register the "DejaVuSans" font, which supports Cyrillic characters
    pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))

    # Create a custom style for your text using the "DejaVuSans" font
    custom_style = ParagraphStyle(
        'CustomStyle',
        parent=styles['Normal'],
        fontName='DejaVuSans',
        fontSize=12,
        textColor=colors.black,
        spaceAfter=12,
    )

    story = []

    # Add your generated text to the story
    for paragraph in text.split('\n'):
        p = Paragraph(paragraph, custom_style)
        story.append(p)

    doc.build(story)
    buffer.seek(0)

    return buffer



def create_tables(cursor):
    commands = [
        '''
        CREATE TABLE IF NOT EXISTS feedback_pdf (
            id SERIAL PRIMARY KEY,
            user_id TEXT,
            rating INT,
            pdf_file BYTEA,
            text TEXT,
            email TEXT
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS history_pdf (
            id SERIAL PRIMARY KEY,
            user_id TEXT,
            pdf_file BYTEA,
            response TEXT
        )
        ''',
    ]

    try:
        for command in commands:
            cursor.execute(command)
            # connection.commit()
    except (Exception, psycopg2.Error) as error:
        print("Problem with SQL:", error)


def establish_database_connection():
    db_host = os.getenv('DB_HOST')
    db_database = os.getenv('DB_DATABASE')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')

    try:
        connection = psycopg2.connect(
            host=db_host,
            database=db_database,
            user=db_user,
            password=db_password
        )
        print("Successfully connected to the PostgreSQL database!")
        return connection
    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL:", error)
        return None



def clear_history():
    st.session_state['pdf-plan']['generated'] = []
    st.session_state['pdf-plan']['names'] = []



def handle_feedback_submission(user_nickname, rating, pdf_content, feedback_input, email):
    pass
    # try:
    #     command = 'INSERT INTO feedback_pdf (user_id, rating, pdf_file, text, email) VALUES(%s, %s, %s, %s, %s)' 
    #     cursor.execute(command, (user_nickname, rating, psycopg2.Binary(pdf_content), feedback_input, email))
    #     connection.commit()
    #     st.success("Feedback submitted successfully!")
    # except (Exception, psycopg2.Error) as error:
    #     print("Error executing SQL statements when setting pdf_file in history_pdf:", error)
    #     connection.rollback()




def print_generated_plans_and_store_in_db():
        for i in range(len(st.session_state['pdf-plan']['generated'])):
            name = st.session_state['pdf-plan']['names'][i]
            with st.expander(name):

                response_for_history = ''
                if source_doc:
                    pdf_for_history = source_doc.read()

                for item in st.session_state['pdf-plan']['generated'][i]:

                    for topic, value in item.items():
                        st.subheader(topic)

                        response_for_history += topic
                        response_for_history += '\n'

                        for inst_speech, content in value.items():
                            st.write(f'{inst_speech} : {content}')
                            st.write()  

                            response_for_history += f'{inst_speech} : {content}'
                            response_for_history += '\n'

                        st.write()
                        response_for_history += '\n'

                # if source_doc:
                #     try:
                #         command = 'INSERT INTO history_pdf (user_id, pdf_file, response) VALUES(%s, %s, %s)' 
                #         cursor.execute(command, (user_nickname, psycopg2.Binary(pdf_for_history), response_for_history,))
                #         connection.commit()
                #     except (Exception, psycopg2.Error) as error:
                #         print("Error executing SQL statements when setting pdf_file in history_pdf:", error)
                #         connection.rollback()
                
                if response_for_history:
                    st.download_button('Загрузить', generate_pdf(response_for_history), file_name=f'{name}.pdf')
                
                st.divider()




if 'pdf-plan' not in st.session_state:
    st.session_state['pdf-plan'] = {
        'generated' : [],
        'names' : []
    }

# connection = establish_database_connection()
# cursor = connection.cursor()

user_nickname = st.text_input("ВВЕДИТЕ ВАШ УНИКАЛЬНЫЙ НИКНЕЙМ ЧТОБ ИСПОЛЬЗОВАТЬ ФУНКЦИЮ 👇")


if user_nickname:
    # create_tables(cursor)

    student_category = st.selectbox(
        'Кому предназначен урок?',
        ('Дети до 6 лет', 'Школьники', 'Cтуденты', 'Взрослые')
    )
    student_level = st.selectbox(
        'Какой уровень у ученика?',
        ('Начинающий', 'Средний', 'Высокий')
    )

    custom_filter = st.text_input("Введите что то еще если есть:")

    st.subheader('Создай сценарии из PDF файла')

    source_doc = st.file_uploader("Загружай свой файл PDF", type="pdf")

    submit_button = st.button(label='Создать')

    if submit_button:
        if not source_doc:
            st.error("Пожалуйста загрузите файл.")
        else:
            file_data = source_doc.read()

            files = {'file': file_data} 

            params = {
                'user_nickname' : user_nickname,
                'student_category': student_category,
                'student_level': student_level,
                'custom_filter': custom_filter
            }

            response = requests.post(url='https://fastapi-ngob.onrender.com/pdf/', params=params, files=files, headers={'accept': 'application/json'})
            
            
            if response.status_code == 200:
                json_data = response.json()
                st.session_state['pdf-plan']['generated'].append(json_data['scenario'])
                st.session_state['pdf-plan']['names'].append(source_doc.name)
            else:
                print(f"Request failed with status code: {response.status_code}")

    clear_button = st.button("Очистить", key="clear")

    st.write("#")

    if clear_button:
        clear_history()


    if st.session_state['pdf-plan']:
        print_generated_plans_and_store_in_db()
    
    st.write("#")

    with st.expander("Отзыв"):
        rating = st.slider('Оцените сервис от 0 до 10', 0, 10, 5)

        bad_pdf = st.file_uploader("Загрузите PDF который не заработал или вывел плохой результат", type="pdf")
        if bad_pdf:
            pdf_content = bad_pdf.read()

        email = st.text_input("ЭЛЕКТРОННАЯ ПОЧТА ДЛЯ ПОЛУЧЕНИЯ ССЫЛКУ НА ФИНАЛЬНЫЙ ПРОДУКТ 👇")

        
        with st.form(key='feedback_form', clear_on_submit=True):
            feedback_input = st.text_area("Что думаете о сервисе? Какие советы или предложения рекомендуете?", key='feedback_input')
            submit_button = st.form_submit_button(label='Отправить')

        if submit_button and feedback_input:

            # handle_feedback_submission(user_nickname, rating, pdf_content, feedback_input, email)

            st.success("Отзыв отправлен!")

# cursor.close()
# connection.close()