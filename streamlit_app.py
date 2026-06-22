from app import (
    extract_video_id,
    get_transcript,
    create_vector_db,
    load_llm,
    get_answer,
    summarize_video,
    generate_notes,
    generate_quiz,
    generate_flashcards
)
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


st.set_page_config(
    page_title="YouTube Chatbot",
    page_icon="🎥",
    layout="wide"
)

st.title("YouTube RAG Chatbot")

st.write(
    "Paste a YouTube URL, process the video, and interact with it."
)

# ==========================================
# INITIALIZE SESSION STATE
# ==========================================

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "transcript" not in st.session_state:
    st.session_state.transcript = None

if "llm" not in st.session_state:
    st.session_state.llm = load_llm()

# ==========================================
# URL INPUT
# ==========================================

url = st.text_input(
    "Paste YouTube URL"
)

# ==========================================
# PROCESS VIDEO
# ==========================================

if st.button("🚀 Process Video"):

    if not url:
        st.error("Please enter a YouTube URL")

    else:

        try:

            with st.spinner(
                "Fetching transcript and creating vector database..."
            ):

                video_id = extract_video_id(url)

                transcript = get_transcript(video_id)

                vectordb = create_vector_db(transcript)

                retriever = vectordb.as_retriever()

                st.session_state.transcript = transcript
                st.session_state.retriever = retriever

            st.success("Video processed successfully!")

        except Exception as e:
            st.error(str(e))

# ==========================================
# IF VIDEO PROCESSED
# ==========================================

if st.session_state.retriever is not None:

    st.divider()

    st.subheader("💬 Ask Questions")

    question = st.text_input(
        "Enter your question"
    )

    if st.button("Get Answer"):

        if question:

            with st.spinner("Generating answer..."):

                answer = get_answer(
                    question,
                    st.session_state.retriever,
                    st.session_state.llm
                )

                st.markdown("### Answer")

                st.write(answer)

    st.divider()

    st.subheader("📚 Study Tools")

    col1, col2 = st.columns(2)

    with col1:

        if st.button("📄 Summarize Video"):

            with st.spinner("Generating summary..."):

                result = summarize_video(
                    st.session_state.transcript,
                    st.session_state.llm
                )

                st.markdown("## 📄 Summary")

                st.write(result)

        if st.button("❓ Generate Quiz"):

            with st.spinner("Generating quiz..."):

                result = generate_quiz(
                    st.session_state.transcript,
                    st.session_state.llm
                )

                st.markdown("## ❓ Quiz")

                st.write(result)

    with col2:

        if st.button("📝 Generate Notes"):

            with st.spinner("Generating notes..."):

                result = generate_notes(
                    st.session_state.transcript,
                    st.session_state.llm
                )

                st.markdown("## 📝 Notes")

                st.write(result)

        if st.button("🧠 Generate Flashcards"):

            with st.spinner("Generating flashcards..."):

                result = generate_flashcards(
                    st.session_state.transcript,
                    st.session_state.llm
                )

                st.markdown("## 🧠 Flashcards")

                st.write(result)


@st.cache_resource
def get_llm():
    return load_llm()


llm = get_llm()
