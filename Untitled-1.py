import json
import os
import tempfile

import psycopg2
import streamlit as st
from dotenv import load_dotenv

from langchain import LLMChain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()

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
            connection.commit()
    except (Exception, psycopg2.Error) as error:
        print("Problem with SQL:", error)


def print_table_rows(table, cursor):
    select_query = f"SELECT * FROM {table}"
    cursor.execute(select_query)
    rows = cursor.fetchall()
    print(f'Rows for {table}')
    for row in rows:
        print(row)


def get_response(user_input, pages):
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5)

    system_prompt = SystemMessagePromptTemplate.from_template(body_plan_template)

    human_template = '''
        Complete the following request: {query}
    '''
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chain_prompt = ChatPromptTemplate.from_messages([human_prompt, system_prompt])

    chain = LLMChain(llm=llm, prompt=chain_prompt, verbose=True)

    with get_openai_callback() as cb:
        response = chain.run(query=user_input, materials=pages)
        print(cb)

    return response


def clear_history():
    st.session_state['pdf-plan']['generated'] = []
    st.session_state['pdf-plan']['past'] = []
    st.session_state['pdf-plan']['messages'] = [
        {"role": "system", "content": "You are a teacher who wants to create a teaching plan."}
    ]


def handle_feedback_submission(user_nickname, rating, pdf_content, feedback_input, email):
    try:
        command = 'INSERT INTO feedback_pdf (user_id, rating, pdf_file, text, email) VALUES(%s, %s, %s, %s, %s)' 
        cursor.execute(command, (user_nickname, rating, psycopg2.Binary(pdf_content), feedback_input, email))
        connection.commit()
        st.success("Feedback submitted successfully!")
    except (Exception, psycopg2.Error) as error:
        print("Error executing SQL statements when setting pdf_file in history_pdf:", error)
        connection.rollback()


def handle_plan_creation(user_nickname, source_doc):
    try:
        with st.spinner('Please wait...'):
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(source_doc.read())
            loader = PyPDFLoader(tmp_file.name)

            pages = loader.load()
            os.remove(tmp_file.name)
            
            responses = []
            size = len(pages) if len(pages) <= 10 else 10

            for i in range(size):
                temp = pages[i].page_content
                response = get_response('Create plan', temp)
                response = json.loads(response)
                responses.append(response)

            response =  responses

            st.session_state['pdf-plan']['past'].append('Create')
            st.session_state['pdf-plan']['generated'].append(response)
    except Exception as e:
        st.exception(f"An error occurred: {e}")


def print_generated_plans():
    for i in range(len(st.session_state['pdf-plan']['generated'])):
        message(st.session_state['pdf-plan']["past"][i], is_user=True, key=str(i) + '_user', avatar_style='no-avatar')

        response_for_history = ''
        pdf_for_history = source_doc.read()
        j = 1
        for item in st.session_state['pdf-plan']['generated'][i]:
            st.header(f'Page {j}')
            j += 1

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

        try:
            command = 'INSERT INTO history_pdf (user_id, pdf_file, response) VALUES(%s, %s, %s)' 
            cursor.execute(command, (user_nickname, psycopg2.Binary(pdf_for_history), response_for_history,))
            connection.commit()
        except (Exception, psycopg2.Error) as error:
            print("Error executing SQL statements when setting pdf_file in history_pdf:", error)
            connection.rollback()


# Main code
load_dotenv()
connection = establish_database_connection()
if connection is None:
    st.error("Failed to connect to the database. Please check your database configuration.")

cursor = connection.cursor()

user_nickname = st.text_input("ВВЕДИТЕ ВАШ УНИКАЛЬНЫЙ НИКНЕЙМ ЧТОБ ИСПОЛЬЗОВАТЬ ФУНКЦИЮ 👇")

if user_nickname:
    create_tables(cursor)

    st.subheader('Создай сценарии из PDF файла')

    source_doc = st.file_uploader("Загружай свой файл PDF", type="pdf")

if 'pdf-plan' not in st.session_state:
    st.session_state['pdf-plan'] = {
        'generated' : [],
        'past' : [],
        'messages' : [
            {"role": "system", "content": "You are a teacher who wants to create a teaching plan."}
        ]
    }

response_container = st.container()
container = st.container()

with container:
    if user_nickname:
        submit_button = st.button(label='Создать')

        if submit_button:
            if not openai_api_key:
                st.error("Please provide the missing API keys in Settings.")
            elif not source_doc:
                st.error("Please provide the lecture document.")
            else:
                with st.spinner('Please wait...'):
                    handle_plan_creation(user_nickname, source_doc)

    if user_nickname:
        clear_button = st.button("Очистить Историю", key="clear")

    st.write("##")
    if user_nickname:
        with st.expander("Форма для отзыва"):
            rating = st.slider('Оцените сервис от 0 до 10', 0, 10, 5)

            bad_pdf = st.file_uploader("Загрузите PDF который не заработал или вывел плохой результат", type="pdf")
            if bad_pdf:
                pdf_content = bad_pdf.read()

            email = st.text_input("ЭЛЕКТРОННАЯ ПОЧТА ДЛЯ ПОЛУЧЕНИЯ ССЫЛКУ НА ФИНАЛЬНЫЙ ПРОДУКТ 👇")

            with st.form(key='feedback_form', clear_on_submit=True):
                feedback_input = st.text_area("Что думаете о сервисе? Какие советы или предложения рекомендуете?", key='feedback_input')
                submit_button = st.form_submit_button(label='Отправить')

            if submit_button and feedback_input:
                handle_feedback_submission(user_nickname, rating, pdf_content, feedback_input, email)

if user_nickname and clear_button:
    clear_history()

if st.session_state['pdf-plan']:
    with response_container:
        print_generated_plans()

cursor.close()
connection.close()