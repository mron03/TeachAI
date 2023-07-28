
import re
import psycopg2, requests, json, os
from dotenv import load_dotenv

from streamlit_tags import st_tags
import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, VideoUnavailable

from langchain import LLMChain
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

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
    You are a teacher, You need to create a teaching scenario for {student_category}

    You are aware that your student knowledge is at {student_level} level, so you adapt the materials to them
    
    For example, if they are beginner, explain them in easy and understanding way. If they are proffient or higher, you can explain in more complex way with good examples if needed

    You need to follow this command {custom_filter}

    You HAVE TO provide examples or problems with solutions if needed for topic explanation

    You need to use the following data to create plan:
        "materials: 
            {materials}
    
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
            connection.commit()
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



def create_plan_by_youtube(prompt, student_category, student_level, custom_filter, yt_urls):
    yt_ids = []
    
    if len(prompt) != 0:
        yt_videos = get_youtube_videos(prompt)
        for video in yt_videos:
            yt_ids.append(video['id']['videoId'])

    for url in yt_urls:
        yt_ids.append(url.split('/')[3])

    docs, videos = split_into_docs(yt_ids)

    llm=ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0, verbose=True)

    system_prompt = SystemMessagePromptTemplate.from_template(body_template)

    human_template = '''
        Complete the following request: {query}
    '''
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chain_prompt = ChatPromptTemplate.from_messages([human_prompt, system_prompt])
    chain = LLMChain(llm=llm, prompt=chain_prompt)
    summarization_chain = load_summarize_chain(llm, chain_type="map_reduce")
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=2000, chunk_overlap=300)


    responses = []
    prev_responses_summary = ''

    with get_openai_callback() as cb:
        for doc in docs:
            r = chain.run(question='create a teaching scenario', query='create a teaching scenario', prev_responses_summary=prev_responses_summary, student_category = student_category, student_level = student_level, custom_filter=custom_filter, materials=doc.page_content)
            responses.append(r)
            
            inp = text_splitter.create_documents(responses)
            prev_responses_summary = summarization_chain.run(inp)
            print('SUMMARIEssssssssssssssssssssssssssssS', prev_responses_summary)
            print(cb)

    return responses, videos
     


def split_into_docs(video_ids):
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0, verbose=True)

    videos = []
    for id in video_ids:
        videos.append(f'https://youtu.be/{id}')

    try:
        transcript_list = YouTubeTranscriptApi.get_transcripts(video_ids, languages=['fr'])
        print('Transcripts are available')
    except TranscriptsDisabled:
        print("Transcripts are disabled for one or more videos.")
    except VideoUnavailable:
        print("One or more videos are unavailable.")
    except Exception as e:
        print("An unexpected error occurred:", e)

    res = ''
    for id in video_ids:
        transcript_list = YouTubeTranscriptApi.list_transcripts(id)
        transcript = transcript_list.find_transcript(['en', 'ru'])
        print(transcript.fetch())
        translated_transcript = transcript.translate('en')
        translated_transcript_fetched = translated_transcript.fetch()

        if len(translated_transcript_fetched) != 0:
            final_transcript = translated_transcript_fetched
        else:
            final_transcript = transcript.fetch()



        for t in final_transcript:
            res = res + t['text'] + '\n'
    
        num_of_tokens = llm.get_num_tokens(res)
        
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=4000, chunk_overlap=500)

        docs = text_splitter.create_documents([res])
        num_docs = len(docs)

        num_tokens_first_doc = llm.get_num_tokens(docs[0].page_content)

        print (f"{num_of_tokens} Now we have {num_docs} documents and the first one has {num_tokens_first_doc} tokens")
         

    
    for d in docs:
        print('DOCUMENTS:')
        print(d.page_content)
        print()
    return docs, videos


def create_db(video_urls):
   
	transcript = ''

	i = 1
   
	for video_url in video_urls:
		loader = YoutubeLoader.from_youtube_url(video_url)
		transcript += f'Youtube video number {i}: '
		i += 1
		transcript += loader.load()

	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size = 1000,
		chunk_overlap = 200,
		length_function = len,
	)

	docs = text_splitter.split_documents(transcript)

	embeddings = OpenAIEmbeddings()
	db = Chroma.from_documents(docs, embeddings)

	return db
   



def get_youtube_videos(prompt):

	url = f"https://www.googleapis.com/youtube/v3/search?key={youtube_api_key}&q={prompt}&type=video&part=snippet&maxResults=1&videoDuration=medium"

	response = requests.get(url)
	data = json.loads(response.content)
	return data["items"]



def print_generated_plans_and_store_in_db():
    with st.expander('Результаты'):
    
        for i in range(len(st.session_state['youtube-plan']['generated'])):
                response_for_history = ''

                for response in st.session_state['youtube-plan']['generated'][i]:
                    print(response)
                    print(type(response))

                for response in st.session_state['youtube-plan']['generated'][i]:

                    for topic, value in response.items():
                        st.subheader(topic)
                        response_for_history += topic
                        response_for_history += '\n'
                            
                        for inst_speech, content in value.items():
                            st.write(f'{inst_speech} : {content}')
                            
                            response_for_history += f'{inst_speech} : {content}'
                            response_for_history += '\n'

                        response_for_history += '\n'
                    

                    # try:
                    #     command = 'INSERT INTO history_youtube (user_id, topic, response) VALUES(%s, %s, %s)' 
                    #     cursor.execute(command, (user_nickname, user_input, response_for_history,))
                    #     connection.commit()

                    # except (Exception, psycopg2.Error) as error:
                    #     print("Error executing SQL statements when setting pdf_file in history_pdf:", error)
                    #     connection.rollback()

                if response_for_history:
                    st.download_button('Загрузить', generate_pdf(response_for_history), 'youtube.pdf')
                
                st.divider()


def is_youtube_link(link):
    # Regular expression pattern to match YouTube video URLs
    youtube_pattern = r'^https?://(?:www\.)?(?:youtu\.be/|youtube\.com/watch\?v=)([\w-]+)'

    # Check if the link matches the YouTube pattern
    return re.match(youtube_pattern, link) is not None


if 'youtube-plan' not in st.session_state:
    st.session_state['youtube-plan'] = {
        'generated' : [],
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

    user_input = st.text_area("Поле для поиска по ютуб", key='input', height=50)
    submit_button = st.button(label='Создать')


    if submit_button and (user_input or yt_urls):

        try:
            with st.spinner('Пожалуйста подождите 2-3 минуты'):
    
                responses, videos = create_plan_by_youtube(user_input, student_category, student_level, custom_filter, yt_urls)
                final_responses = []
                for response in responses:
                    final_responses.append(json.loads(response))

                st.session_state['youtube-plan']['generated'].append(final_responses)

        except Exception as e:
            st.exception(f"An error occurred: {e}")



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