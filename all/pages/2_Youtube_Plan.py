
import re
import psycopg2, requests, json, os
from dotenv import load_dotenv

from streamlit_tags import st_tags
import streamlit as st


from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO


body_template = '''
    Based on PREVIOUS RESPONSES SUMMARY write the LOGICAL CONTINUATION OF THE SCENARIO, DO NOT REPEAT THE CONTENT:
        ```
            {prev_responses_summary}
        ```

    You need to use the following data to create plan:
            ```{materials}```

    You are a teacher, You need to create a teaching scenario for {student_category}

    You are aware that your student knowledge is at {student_level} level, so you adapt the materials to them
    
    For example, if they are beginner, explain them in easy and understanding way. If they are proffient or higher, you can explain in more complex way with good examples if needed

    You need to follow this command ```{custom_filter}```


    Return the answer in VALID JSON format in russian language:
        {{
            "Write the topic name" : {{
                "Instruction 1" : "Write What to do",
                "Speech 1": "Write what to tell for instruction 1",
                "Instruction 2" : "Write What to do",
                "Speech 2": "Write what to tell for instruction 2",
                "Instruction 3" : "Write What to do",
                "Speech 3": "Write what to tell for instruction 3",
                "Instruction N": "...",
                "Speech N": "..."
            }}
        }}
    
    Example of idiomatic JSON response: {{"Integrals":{{"Instruction 1":"Introduce topic of integrals","Speech 1":"Today, we are going to learn integrals","Instruction 2":"Show examples and problems","Speech 2":"Here is the problem we are going to solve","Instruction 3":"Conclude the topic","Speech 3":"In conclusion, integrals are very useful"}}}}
'''

load_dotenv()
youtube_api_key = os.getenv('YOUTUBE_API_KEY')
openai_api_key = os.getenv("OPENAI_API_KEY")

def generate_pdf(text):
    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))

    custom_style = ParagraphStyle(
        'CustomStyle',
        parent=styles['Normal'],
        fontName='DejaVuSans',
        fontSize=12,
        textColor=colors.black,
        spaceAfter=12,
    )

    story = []

    for paragraph in text.split('\n'):
        p = Paragraph(paragraph, custom_style)
        story.append(p)

    doc.build(story)
    buffer.seek(0)

    return buffer

def create_tables(cursor):
    commands = [
        '''
        CREATE TABLE IF NOT EXISTS feedback_youtube (
            id SERIAL PRIMARY KEY,
            user_id TEXT,
            rating INT,
            text TEXT,
            email TEXT
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS history_youtube (
            id SERIAL PRIMARY KEY,
            user_id TEXT,
            topic TEXT,
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
    st.session_state['youtube-plan']['generated'] = []


def print_generated_plans_and_store_in_db():    
    for i in range(len(st.session_state['youtube-plan']['generated'])):
            response_for_history = ''

            for response in st.session_state['youtube-plan']['generated'][i]:
                for r in response:
                    for topic, value in json.loads(r).items():
                        with st.expander(topic):

                            st.subheader(topic)
                            response_for_history += topic
                            response_for_history += '\n'
                                
                            for inst_speech, content in value.items():
                                st.write(f'{inst_speech} : {content}')
                                
                                response_for_history += f'{inst_speech} : {content}'
                                response_for_history += '\n'

                            response_for_history += '\n'
                            if response_for_history:
                                st.download_button('Загрузить', generate_pdf(response_for_history), 'youtube.pdf')
                    

                    # try:
                    #     command = 'INSERT INTO history_youtube (user_id, topic, response) VALUES(%s, %s, %s)' 
                    #     cursor.execute(command, (user_nickname, user_input, response_for_history,))
                    #     connection.commit()

                    # except (Exception, psycopg2.Error) as error:
                    #     print("Error executing SQL statements when setting pdf_file in history_pdf:", error)
                    #     connection.rollback()
            
            
    
    if st.session_state['youtube-plan']['no_transcript_urls']:
        st.subheader('Эти ссылки не имеют транскрипта: \n')
        for url in st.session_state['youtube-plan']['no_transcript_urls']:
            st.write(url[0])



def is_youtube_link(link):
    youtube_pattern = r'^https?://(?:www\.)?(?:youtu\.be/|youtube\.com/watch\?v=)([\w-]+)'
    return re.match(youtube_pattern, link) is not None


if 'youtube-plan' not in st.session_state:
    st.session_state['youtube-plan'] = {
        'generated' : [],
        'no_transcript_urls' : []
    }


# connection = establish_database_connection()
# cursor = connection.cursor()

user_nickname = st.text_input("ВВЕДИТЕ ВАШ УНИКАЛЬНЫЙ НИКНЕЙМ ЧТОБ ИСПОЛЬЗОВАТЬ ФУНКЦИЮ 👇")
if user_nickname:
    # create_tables(cursor)

    st.subheader('Создай план используя ютуб')

    student_category = st.selectbox(
        'Кому предназначен урок?',
        ('Дети до 6 лет', 'Школьники', 'Cтуденты', 'Взрослые')
    
    )
    student_level = st.selectbox(
        'Какой уровень у ученика?',
        ('Начинающий', 'Средний', 'Высокий')
    
    )

    custom_filter = st.text_input("Введите что то еще если есть:")

    yt_urls = st_tags(
        label='Поле для ссылки ютуб видео:',
        text='Нажмите Enter чтоб добавить',
    )

    for url in yt_urls:
        if not is_youtube_link(url):
            st.error(f'Invalid url {url}')

    youtube_prompt = st.text_area("Поле для поиска по ютуб", key='input', height=50)
    submit_button = st.button(label='Создать')


    if submit_button and (youtube_prompt or yt_urls):

        try:
            with st.spinner('Пожалуйста подождите 2-3 минуты'):

                data = {
                    'user_nickname' : user_nickname,
                    'youtube_urls' : yt_urls,
                    'youtube_prompt' : youtube_prompt,
                    'student_category': student_category,
                    'student_level': student_level,
                    'custom_filter': custom_filter
                }

                response = requests.post(url='https://fastapi-ngob.onrender.com/youtube/create-scenario', json=data, headers={'accept': 'application/json', 'Content-Type': 'application/json'})
            
                if response.status_code == 200:
                    json_data = response.json()
                    st.session_state['youtube-plan']['generated'].append(json_data['scenario'][:-1])
                    st.session_state['youtube-plan']['no_transcript_urls'].append(json_data['scenario'][-1])
                else:
                    print(f"Request failed with status code: {response.status_code}")

        except Exception as e:
            st.exception('Exception: ', e)


    clear_button = st.button("Очистить", key="clear")

    if st.session_state['youtube-plan']:
        print_generated_plans_and_store_in_db()

    with st.expander("Форма для отзыва"):

        rating = st.slider('Оцените сервис от 0 до 10', 0, 10, 5)
        
        email = st.text_input("ЭЛЕКТРОННАЯ ПОЧТА ДЛЯ ПОЛУЧЕНИЯ ССЫЛКУ НА ФИНАЛЬНЫЙ ПРОДУКТ 👇")
        
        with st.form(key='feedback_form', clear_on_submit=True):
            feedback_input = st.text_area("Что думаете о сервисе? Какие советы или предложения рекомендуете?", key='feedback_input')
            submit_button = st.form_submit_button(label='Отправить')
            

        if submit_button and feedback_input:

            # try:
            #     command = 'INSERT INTO feedback_youtube (user_id, rating, text, email) VALUES(%s, %s, %s, %s)' 
            #     cursor.execute(command, (user_nickname, rating, feedback_input, email))
            #     connection.commit()

            # except (Exception, psycopg2.Error) as error:
            #     print("Error executing SQL statements when setting pdf_file in history_pdf:", error)
            #     connection.rollback()

            st.success("Feedback submitted successfully!")

    if clear_button:
        clear_history()


# cursor.close()
# connection.close()