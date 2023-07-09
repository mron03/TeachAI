from dotenv import load_dotenv
from langchain import FAISS, OpenAI, PromptTemplate
import requests, json, os
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain


load_dotenv()
youtube_api_key = os.getenv('YOUTUBE_API_KEY')
openai_api_key = os.getenv("OPENAI_API_KEY")

def create_plan_by_youtube(prompt):

    yt_videos = get_youtube_videos(prompt)
    yt_urls = []

    for video in yt_videos:
        video_id = video['id']['videoId']
        yt_urls.append(f'https://youtu.be/{video_id}')

    db = create_db(yt_urls)
    search = db.asimilarity_search('Create a teaching plan about integral')

    llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.4, openai_api_key=openai_api_key)

    chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

    with get_openai_callback() as cb:
        response = chain.run(question='Create a teaching plan about integral', input_documents=search, llm=llm)

        print(cb)

    return response


def create_db(video_urls):

	transcripts = []

	for video_url in video_urls:
		loader = YoutubeLoader.from_youtube_url(video_url)
		transcripts.extend(loader.load())
	
	print(transcripts)

	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size = 1000,
		chunk_overlap = 200,
		length_function = len,
	)

	docs = ''
	i = 1
	for doc in transcripts:
		docs += f'Youtube video number {i}:'
		i += 1
		docs += text_splitter.split_documents(doc.page_content)
	
	

	embeddings = OpenAIEmbeddings()
	db = FAISS.from_documents(docs, embeddings)

	return db




def get_youtube_videos(prompt):

	url = f"https://www.googleapis.com/youtube/v3/search?key={youtube_api_key}&q={prompt}&type=video&part=snippet&maxResults=1"

	response = requests.get(url)
	data = json.loads(response.content)
	return data["items"]


print(create_plan_by_youtube('What is integral in calculus?'))

