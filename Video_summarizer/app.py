import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file,get_file
import google.generativeai as genai

import time
from pathlib import Path

import tempfile

from dotenv import load_dotenv
load_dotenv()

import os

API_KEY=os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

st.title("Phidata video AI summarizer Agent")
st.header('powered by Gemini 2.0 Flash Exp ')

@st.cache_resource
def initialize_agent():
    return Agent(
        name="video AI summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )
## Initialize the agent
multimodal_Agent = initialize_agent()

video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"],help="Upload a video for AI analysis")

if video_file:
    with tempfile.NamedTemporaryFile(delete=False,suffix='.mp4') as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    st.video(video_path,format='video/mp4',start_time=0)

    user_query=st.text_area(
        "what insights are you seeking from this video?",
        placeholder="Ask anything about the video content. The AI agent will analyze and gather additional information",
        help="provide specific questions or insights you want to know about the video content"
    )

    if st.button("Analyze video",key="analyze_video_button"):
        if not user_query:
            st.warning("Please provide a query to analyze the video content")
        else:
            try:
                with st.spinner("processing video and gathering insights..."):
                    #upload and process the video
                    processed_video = upload_file(video_path)
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)

                    # prompt generation for analysis
                    analysis_prompt = (
                        f"""Analyze the uploaded video for content and context.
                         Respond to the following query using video insights and supplementary web research
                         {user_query} 
                         provide a detailed,user-friendly and actionable response."""
                         )
                    # AI agent processing
                    response=multimodal_Agent.run(analysis_prompt,videos=[processed_video])
                # Display the response
                st.subheader("Analysis Result")
                st.markdown(response.content)
            except Exception as error:
                st.error(f"An error occurred during video analysis: {error}")
            finally:
                Path(video_path).unlink(missing_ok=True)
else:
    st.info('upload a video file to analyze its content')

    #customize textarea
st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)