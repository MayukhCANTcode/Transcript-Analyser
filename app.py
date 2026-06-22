from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


# ==========================
# EXTRACT VIDEO ID
# ==========================
def extract_video_id(url):

    if "watch?v=" in url:
        return url.split("v=")[1].split("&")[0]

    elif "shorts/" in url:
        return url.split("shorts/")[1].split("?")[0]

    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]

    else:
        raise ValueError("Unsupported YouTube URL")


# ==========================
# GET TRANSCRIPT
# ==========================
def get_transcript(video_id):

    ytt_api = YouTubeTranscriptApi()

    transcript = ytt_api.fetch(video_id)

    full_text = ""

    for item in transcript:
        full_text += item.text + " "

    return full_text


# ==========================
# CREATE VECTOR DB
# ==========================
def create_vector_db(text):
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,

    )

    return vectordb


# ==========================
# LOAD LLM
# ==========================
def load_llm():
    api_key = os.getenv("GOOGLE_API_KEY")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key
    )

    return llm


# ==========================
# QUESTION ANSWERING (RAG)
# ==========================
def get_answer(question, retriever, llm):

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""
Answer ONLY from the provided context.

If the answer is not available in the context,
say so.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content


# ==========================
# VIDEO SUMMARY
# ==========================
def summarize_video(transcript, llm):

    prompt = f"""
Create a detailed summary of the following video.

Transcript:
{transcript}
"""

    response = llm.invoke(prompt)

    return response.content


# ==========================
# STUDY NOTES
# ==========================
def generate_notes(transcript, llm):

    prompt = f"""
Create detailed study notes.

Requirements:
- Use headings
- Use bullet points
- Include important concepts
- Include key takeaways

Transcript:
{transcript}
"""

    response = llm.invoke(prompt)

    return response.content


# ==========================
# QUIZ
# ==========================
def generate_quiz(transcript, llm):

    prompt = f"""
Generate 10 quiz questions with answers
from the following transcript.

Transcript:
{transcript}
"""

    response = llm.invoke(prompt)

    return response.content


# ==========================
# FLASHCARDS
# ==========================
def generate_flashcards(transcript, llm):

    prompt = f"""
Create flashcards.

Format:

Q:
A:

Transcript:
{transcript}
"""

    response = llm.invoke(prompt)

    return response.content
