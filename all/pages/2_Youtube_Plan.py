import psycopg2, requests, json, os
from dotenv import load_dotenv

from streamlit_chat import message
import streamlit as st

from langchain import LLMChain, OpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)



body_template = '''
    You are a teacher who wants to create a very detailed teaching plan with full explanation of every concept to his loving student

    You have to provide examples or problems with solutions if needed for topic explanation

    You need to use the following data to create plan:
        "materials: 
            {materials}
    
    Write the full explanatory speech of each instruction under each instruction part

    DO NOT RETURN YOUR ANSWER TWICE, KEEP THE ANSWER UNIQUE WITHOUT DUPLICATES

    Return the answer strictly like this JSON format:

        "Write the topic or subtopic name or something that makes sense" : {{ 

            "Instruction 1" : "Write What to do"
            "Speech 1": 
                "Write what to tell for instruction 1"
            
            "Instruction 2" : Write What to do 
            "Speech 2": 
                "Write what to tell for instruction 2"

            "Instruction 3" : Write What to do 
            "Speech 3": 
                "Write what to tell for instruction 3"

        }}
'''

load_dotenv()
youtube_api_key = os.getenv('YOUTUBE_API_KEY')
openai_api_key = os.getenv("OPENAI_API_KEY")


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
    st.session_state['youtube-plan']['past'] = []
    st.session_state['youtube-plan']['messages'] = [
        {"role": "system", "content": "You are a teacher who wants to create a teaching plan based on youtube videos."}
    ]


def create_plan_by_youtube(prompt):
   
    yt_videos = get_youtube_videos(prompt)
    yt_urls = []

    for video in yt_videos:
        video_id = video['id']['videoId']
        yt_urls.append(f'https://youtu.be/{video_id}')

    # db = create_db(yt_urls)
    summaries = summarize_videos(yt_urls)

    llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5)

    system_prompt = SystemMessagePromptTemplate.from_template(body_template)

    human_template = '''
        Complete the following request: {query}
    '''
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chain_prompt = ChatPromptTemplate.from_messages([human_prompt, system_prompt])

    chain = LLMChain(llm=llm, prompt=chain_prompt)

    materials = ''

    for summary in summaries:
         materials += summary
         materials += '\n'

    # search = vectordb.similarity_search(user_input)
    with get_openai_callback() as cb:
        response = chain.run(question='create a plan', query='create a teaching plan', materials=materials)
        print(cb)

    return response
     

def summarize_videos(video_urls):
    summaries = []
    prompt_template = """Write VERY DETAILED SUMMARY with SHORT SPEECH OF THE CONTENT of the following in bullet points:
        {text}

    DETAILED SUMMARY:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm=OpenAI(), chain_type="stuff", prompt=prompt, verbose=True)

    for url in video_urls:
         loader = YoutubeLoader.from_youtube_url(youtube_url=url)
         transcript = loader.load_and_split()
         summary = chain.run(transcript)
         summaries.append(summary)

    return summaries
    


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

	url = f"https://www.googleapis.com/youtube/v3/search?key={youtube_api_key}&q={prompt}&type=video&part=snippet&maxResults=4&videoDuration=medium"

	response = requests.get(url)
	data = json.loads(response.content)
	return data["items"]



def print_generated_plans_and_store_in_db():
    for i in range(len(st.session_state['youtube-plan']['generated'])):
            message(st.session_state['youtube-plan']["past"][i], is_user=True, key=str(i) + '_user', avatar_style='no-avatar')

            response_for_history = ''

            for topic, value in st.session_state['youtube-plan']['generated'][i].items():
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
                command = 'INSERT INTO history_youtube (user_id, topic, response) VALUES(%s, %s, %s)' 
                cursor.execute(command, (user_nickname, user_input, response_for_history,))
                connection.commit()

            except (Exception, psycopg2.Error) as error:
                print("Error executing SQL statements when setting pdf_file in history_pdf:", error)
                connection.rollback()



if 'youtube-plan' not in st.session_state:
    st.session_state['youtube-plan'] = {
        'generated' : [],
        'past' : [],
        'messages' : [
            {"role": "system", "content": "You are a teacher who wants to create a teaching plan based on Youtube videos."}
        ]
    }


connection = establish_database_connection()
cursor = connection.cursor()


user_nickname = st.text_input("ВВЕДИТЕ ВАШ УНИКАЛЬНЫЙ НИКНЕЙМ ЧТОБ ИСПОЛЬЗОВАТЬ ФУНКЦИЮ 👇")
if user_nickname:
    create_tables(cursor)

    st.subheader('Создай план используя ютуб')


response_container = st.container()
container = st.container()

with container:
    if user_nickname:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("Введи название темы", key='input', height=50)
            submit_button = st.form_submit_button(label='Отправить')


        if submit_button and user_input:
            if not openai_api_key:
                st.error("Please provide the missing API keys in Settings.")
            else:
                try:
                    with st.spinner('Please wait...'):
            
                        response = json.loads(create_plan_by_youtube(user_input))
                        st.session_state['youtube-plan']['past'].append(user_input)
                        st.session_state['youtube-plan']['generated'].append(response)

                except Exception as e:
                    st.exception(f"An error occurred: {e}")



        clear_button = st.button("Очистить историю", key="clear")

        st.write('#')

        with st.expander("Форма для отзыва"):

            rating = st.slider('Оцените сервис от 0 до 10', 0, 10, 5)
            
            email = st.text_input("ЭЛЕКТРОННАЯ ПОЧТА ДЛЯ ПОЛУЧЕНИЯ ССЫЛКУ НА ФИНАЛЬНЫЙ ПРОДУКТ 👇")
            
            with st.form(key='feedback_form', clear_on_submit=True):
                feedback_input = st.text_area("Что думаете о сервисе? Какие советы или предложения рекомендуете?", key='feedback_input')
                submit_button = st.form_submit_button(label='Отправить')
                

            if submit_button and feedback_input:

                try:
                    command = 'INSERT INTO feedback_youtube (user_id, rating, text, email) VALUES(%s, %s, %s, %s)' 
                    cursor.execute(command, (user_nickname, rating, feedback_input, email))
                    connection.commit()

                except (Exception, psycopg2.Error) as error:
                    print("Error executing SQL statements when setting pdf_file in history_pdf:", error)
                    connection.rollback()

                st.success("Feedback submitted successfully!")

        if clear_button:
            clear_history()

if st.session_state['youtube-plan']:
    with response_container:
        print_generated_plans_and_store_in_db()

cursor.close()
connection.close()
