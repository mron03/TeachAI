import json
import os, tempfile
from dotenv import load_dotenv
from langchain import LLMChain
import psycopg2
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


try:
    connection = psycopg2.connect(
        host="172.22.0.2",
        port="5432",
        database="postgres",
        user="mm",
        password="mm"
    )
    print("Successfully connected to the PostgreSQL database!")
except (Exception, psycopg2.Error) as error:
    print("Error while connecting to PostgreSQL:", error)

human_template = '''Complete the following request: {query}'''

body_plan_template = '''
    You are a teacher who wants to create a very detailed teaching plan with full explanation of every concept to his loving student

    You have to provide examples or problems with solutions if needed for topic explanation

    You have to return the answer in JSON FORMAT

    You need to use the following data to create plan:
        "materials": 
            {materials}
    

    DO NOT RETURN YOUR ANSWER TWICE, KEEP THE ANSWER UNIQUE WITHOUT DUPLICATES

    Return the answer strictly like this JSON format::
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
    RETURN THE ANSWER IN JSON FORMAT

    
'''

def get_response(user_input, pages):
    llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5)

    system_prompt = SystemMessagePromptTemplate.from_template(body_plan_template)

    human_template = '''
        Complete the following request: {query}
    '''
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chain_prompt = ChatPromptTemplate.from_messages([human_prompt, system_prompt])


    chain = LLMChain(llm=llm, prompt=chain_prompt, verbose=True)

    # search = vectordb.similarity_search(user_input)
    with get_openai_callback() as cb:
        response = chain.run(query=user_input, materials=pages)
        print(cb)

    return response


# def correct_result(result):
#     llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.1)

#     system_template = ''''
#         You are a assistant to complete the given teaching plan

#         You need to use this sample plan:
#             {result}

#         Write the answer in this format:
#             {example} 
#     '''

#     system_prompt = SystemMessagePromptTemplate.from_template(system_template)

#     human_template = '''
#         Complete the teaching plan 
#     '''
#     human_prompt = HumanMessagePromptTemplate.from_template(human_template)

#     chain_prompt = ChatPromptTemplate.from_messages([human_prompt, system_prompt])

#     chain = LLMChain(llm=llm, prompt=chain_prompt)

#     # search = vectordb.similarity_search(user_input)
#     with get_openai_callback() as cb:
#         response = chain.run(question='create', result=result, example=example_main)
#         print(cb)

#     return response


load_dotenv()
# Set API keys from session state
# openai_api_key = st.session_state.openai_api_key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit app
st.subheader('Build Plan by PDF')
source_doc = st.file_uploader("Upload Lecture Document", type="pdf")


if 'pdf-plan' not in st.session_state:
    st.session_state['pdf-plan'] = {
        'generated' : [],
        'past' : [],
        'messages' : [
            {"role": "system", "content": "You are a teacher who wants to create a teaching plan."}
        ]
    }


# # Initialise session state variables
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

#     with open('transcript.txt', "r") as f:
#         transcript = f.read()

#     # response = ai.update_transcript(prompt, transcript)
#     response = '123'
#     with open('transcript.txt', "w") as f:
#         f.write(response)
    
#     # response = ai.analyzer_transcript(prompt, transcript)

#     st.session_state['messages'].append({"role": "assistant", "content": response})

#     return response


# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    submit_button = st.button(label='Create')


    # with st.form(key='my_form', clear_on_submit=True):
    #     # user_input = st.text_area("You:", key='input', height=50)
    #     submit_button = st.form_submit_button(label='Create')


    # user_input = st.text_input('You')

    if submit_button:

        # Validate inputs
        if not openai_api_key:
            st.error("Please provide the missing API keys in Settings.")
        elif not source_doc:
            st.error("Please provide the lecture document.")
        else:
            try:
                with st.spinner('Please wait...'):
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(source_doc.read())
                    loader = PyPDFLoader(tmp_file.name)

                    text_splitter = RecursiveCharacterTextSplitter(
                        separators = '\n',
                        chunk_size = 500,
                        chunk_overlap = 150,
                        length_function = len
                    )

                    pages = loader.load()
                    os.remove(tmp_file.name)
                    
                    responses = []
                    for p in pages:
                        temp = text_splitter.split_text(p.page_content)
                        response = get_response('Create plan', temp)
                        response = json.loads(response)
                        responses.append(response)

                    response =  responses

                    st.session_state['pdf-plan']['past'].append('Create')
                    st.session_state['pdf-plan']['generated'].append(response)
            except Exception as e:
                st.exception(f"An error occurred: {e}")



    clear_button = st.button("Clear", key="clear")

    st.write("##")
    st.subheader("Feedback Form")
    with st.form(key='feedback_form', clear_on_submit=True):
        feedback_input = st.text_area("You:", key='feedback_input', height=50)
        submit_button = st.form_submit_button(label='Send')

    if submit_button:
        st.success("Feedback submitted successfully!")


if clear_button:
    st.session_state['pdf-plan']['generated'] = []
    st.session_state['pdf-plan']['past'] = []
    st.session_state['pdf-plan']['messages'] = [
        {"role": "system", "content": "You are a teacher who wants to create a teaching plan."}
    ]

if st.session_state['pdf-plan']:
    with response_container:
        for i in range(len(st.session_state['pdf-plan']['generated'])):
            message(st.session_state['pdf-plan']["past"][i], is_user=True, key=str(i) + '_user', avatar_style='no-avatar')
            # message(st.session_state["generated"][i], key=str(i))
            # st.success(st.session_state['pdf-plan']["generated"][i])
           
            j = 1
            for item in st.session_state['pdf-plan']['generated'][i]:
                st.header(f'Page {j}')
                j += 1

                for topic, value in item.items():
                    st.subheader(topic)
                    
                    for inst_speech, content in value.items():
                        st.write(f'{inst_speech} : {content}')
                        st.write()
                    
                    st.write()

    



