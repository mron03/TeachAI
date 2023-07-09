from dotenv import load_dotenv
from langchain import LLMChain, OpenAI
import requests, json, os
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


load_dotenv()
youtube_api_key = os.getenv('YOUTUBE_API_KEY')
openai_api_key = os.getenv("OPENAI_API_KEY")

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
    chain = load_summarize_chain(llm=OpenAI(), chain_type="stuff", prompt=prompt)

    for url in video_urls:
         loader = YoutubeLoader.from_youtube_url(youtube_url=url)
         transcript = loader.load_and_split()
         summary = chain.run(transcript)
         summaries.append(summary)
        #  print(url)
        #  print(summary)

        #  print('\n\n')

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

	url = f"https://www.googleapis.com/youtube/v3/search?key={youtube_api_key}&q={prompt}&type=video&part=snippet&maxResults=2&videoDuration=medium"

	response = requests.get(url)
	data = json.loads(response.content)
	return data["items"]




# def test(prompt):
#      yt_videos = get_youtube_videos(prompt)
#      yt_urls = []

#      for video in yt_videos:
#         video_id = video['id']['videoId']
#         yt_urls.append(f'https://youtu.be/{video_id}')
    
#      summaries = summarize_videos(yt_urls)



import os, tempfile
from dotenv import load_dotenv
from langchain import PromptTemplate
import streamlit as st
from langchain.llms.openai import OpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback





load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


st.subheader('Build Plan By Youtube')

if 'youtube-plan' not in st.session_state:
    st.session_state['youtube-plan'] = {
        'generated' : [],
        'past' : [],
        'messages' : [
            {"role": "system", "content": "You are a teacher who wants to create a teaching plan based on Youtube videos."}
        ]
    }

# if 'generated' not in st.session_state:
#     st.session_state['generated'] = []
# if 'past' not in st.session_state:
#     st.session_state['past'] = []
# if 'messages' not in st.session_state:
#     st.session_state['messages'] = [
#         {"role": "system", "content": "You are a helpful assistant."}
#     ]



# # generate a response
# def generate_response(prompt):
#     st.session_state['messages'].append({"role": "user", "content": prompt})


#     response = '123'


#     st.session_state['messages'].append({"role": "assistant", "content": response})

#     return response


# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=50)
        submit_button = st.form_submit_button(label='Send')


    # user_input = st.text_input('You')

    if submit_button and user_input:

        # Validate inputs
        if not openai_api_key:
            st.error("Please provide the missing API keys in Settings.")
        else:
            try:
                with st.spinner('Please wait...'):

                    # response = ai.create_plan_by_youtube()
           
                    response = json.loads(create_plan_by_youtube(user_input))
                    print(type(response))
                    print(response)
                    st.session_state['youtube-plan']['past'].append(user_input)
                    st.session_state['youtube-plan']['generated'].append(response)

            except Exception as e:
                st.exception(f"An error occurred: {e}")



    clear_button = st.button("Clear", key="clear")

    st.write('#')
    st.subheader("Feedback Form")
    with st.form(key='feedback_form', clear_on_submit=True):
        feedback_input = st.text_area("You:", key='feedback_input', height=20)
        submit_button = st.form_submit_button(label='Send')

    if submit_button:

        st.success("Feedback submitted successfully!")



if clear_button:
    st.session_state['youtube-plan']['generated'] = []
    st.session_state['youtube-plan']['past'] = []
    st.session_state['youtube-plan']['messages'] = [
        {"role": "system", "content": "You are a teacher who wants to create a teaching plan based on youtube videos."}
    ]




if st.session_state['youtube-plan']:
    with response_container:
        for i in range(len(st.session_state['youtube-plan']['generated'])):
            message(st.session_state['youtube-plan']["past"][i], is_user=True, key=str(i) + '_user', avatar_style='no-avatar')
            # message(st.session_state['youtube-plan']["generated"][i], key=str(i))

            
            for topic, value in st.session_state['youtube-plan']['generated'][i].items():
                st.subheader(topic)
                    
                for inst_speech, content in value.items():
                    st.write(f'{inst_speech} : {content}')
                    st.write()
                
                st.write()