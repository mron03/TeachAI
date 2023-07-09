import json
import os, tempfile
import textwrap
from PyPDF2 import DocumentInformation
from dotenv import load_dotenv
from langchain import LLMChain, PromptTemplate
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
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.document_loaders import TextLoader
import psycopg2
from langchain.memory import ConversationSummaryMemory


















# try:
#     connection = psycopg2.connect(
#         host="127.0.0.1",
#         port="5432",
#         database="postgres",
#         user="postgres",
#         password="changeme"
#     )
#     print("Successfully connected to the PostgreSQL database!")
# except (Exception, psycopg2.Error) as error:
#     print("Error while connecting to PostgreSQL:", error)




example_main = '''
    Teaching Plan: Memory Basics

    Objective: To provide an understanding of memory definitions, types, and organization in microcontrollers.

    Introduction:

        Welcome students and introduce the topic of memory basics in microcontrollers.
        Explain the importance of memory in microcontrollers and its role in storing and retrieving data.
            Speech: "Good morning/afternoon, everyone. 
            Today, we will be diving into the fascinating world of memory basics in microcontrollers. 
            Memory plays a crucial role in storing and retrieving data in microcontrollers, making it an essential topic to understand.

    Main Content:

    I. Memory Definitions

        Define memory as a collection of storage cells with circuits to transfer information.
        Speech: 
            Let's begin by defining memory. Memory is a collection of storage cells along with the necessary circuits to transfer 
            information to and from them. It provides a means for a microcontroller to store and retrieve data. Memory organization 
        
        Explain memory organization as the architectural structure for accessing data.
        Speech:
            Memory organization 
            refers to the architectural structure of memory and how data is accessed. One type of memory we will explore is Random Access Memory (RAM).

        Introduce Random Access Memory (RAM) as a memory organized to transfer data to or from any cell.
        Speech:
            One type of memory we will explore is Random Access Memory (RAM).
            RAM is a memory organized in such a way that data can be transferred to or from any cell or 
            collection of cells without being dependent on the specific cell selected.
            This allows for efficient data access and manipulation.


    II. Typical Data Elements

        Intruction 1: Explain different data elements, such as bits, bytes, and words.
        Speech: 
            Now, let's delve into the typical data elements found in memory. 

        Intruction 2: Define a bit as a single binary digit.
        Speech: 
            A bit is the smallest unit of data in memory, representing a single binary digit.

        Intruction 3: Define a byte as a collection of eight bits accessed together.
        Speech:
            A byte, on the other hand, is a collection of eight bits accessed together. It is a fundamental unit of memory storage.

        Intruction 4: Define a word as a collection of binary bits, typically a power of two multiple of bytes.
        Speech:
            It is usually a power of two multiple of bytes, such as 1 byte, 2 bytes, 4 bytes, or 8 bytes.

    III. Memory Operations

        Intruction 1: Discuss memory operations supported by the memory unit.
        Speech: 
            Memory operations involve reading from and writing to memory data elements. T

        Intruction 2: Explain read and write operations on memory data elements.
        Speech: 
            These operations are supported by the memory unit. 

        Intruction 3: Provide examples of read and write operations on bits, bytes, and words.
        Speech: 
            For example, we can read or write a bit, byte, or word from memory. 

    Conclusion

        Intruction 1: Summarize the key points discussed in the teaching session.
        Intruction 2: Emphasize the importance of understanding memory basics in microcontrollers.
        Intruction 3: Encourage students to explore further resources and practice applying memory concepts.
        
        Speech:
            To wrap up, understanding memory basics is crucial for working with microcontrollers. We have covered 
            memory definitions, types, and organization, as well as memory operations. I encourage you to explore 
            further resources and practice applying these concepts in your projects.

'''

full_plan_system_template = '''
    You are a teacher who wants to create a very detailed teaching plan with full explanation of every concept to his loving student

    You will write a full speech like you need to present the lecture before the students

    You need to use the following data:
        {materials}
    
    Write the full explanatory speech of each instruction under each instruction part

    Return the answer in this format and make it professional:
        The example of format:
            {example}
'''
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

    memory = ConversationSummaryMemory(llm=llm)

    chain = LLMChain(llm=llm, prompt=chain_prompt, memory=memory, verbose=True)

    # search = vectordb.similarity_search(user_input)
    with get_openai_callback() as cb:
        response = chain.run(query=user_input, materials=pages)
        print(cb)

    return response


def correct_result(result):
    llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.1)

    system_template = ''''
        You are a assistant to complete the given teaching plan

        You need to use this sample plan:
            {result}

        Write the answer in this format:
            {example} 
    '''

    system_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = '''
        Complete the teaching plan 
    '''
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chain_prompt = ChatPromptTemplate.from_messages([human_prompt, system_prompt])

    chain = LLMChain(llm=llm, prompt=chain_prompt)

    # search = vectordb.similarity_search(user_input)
    with get_openai_callback() as cb:
        response = chain.run(question='create', result=result, example=example_main)
        print(cb)

    return response


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

    



