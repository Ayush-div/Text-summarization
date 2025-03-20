import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import os
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")
    st.write("Select language for YouTube transcripts (if applicable):")
    language_option = st.selectbox(
        "Transcript Language",
        options=["Auto", "English", "Hindi", "Spanish", "French", "German"],
        index=0
    )
    
    language_map = {
        "Auto": None,
        "English": "en",
        "Hindi": "hi",
        "Spanish": "es",
        "French": "fr",
        "German": "de"
    }
    
    selected_language = language_map[language_option]


generic_url = st.text_input("URL", label_visibility="collapsed")


def initialize_llm(api_key):
    return ChatGroq(model="llama3-8b-8192", groq_api_key=api_key or os.getenv("GROQ_API_KEY"))


map_prompt_template = """
Summarize the following part of a transcript in 100 words:
{text}
"""
map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

combine_prompt_template = """
The following is a set of summaries from different parts of a transcript.
Combine them into a coherent summary of 300 words:
{text}
"""
combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

stuff_prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
stuff_prompt = PromptTemplate(template=stuff_prompt_template, input_variables=["text"])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=200,
    length_function=len
)

def extract_youtube_id(url):
    parsed_url = urlparse(url)
    if parsed_url.netloc == 'youtu.be':
        return parsed_url.path.lstrip('/')
    elif parsed_url.netloc in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query).get('v', [None])[0]
        elif parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/')[2]
        elif parsed_url.path.startswith('/v/'):
            return parsed_url.path.split('/')[2]
    return None

# Function to process YouTube content
def process_youtube(video_id, llm, language_code=None):
    st.info("Processing YouTube video...")
    
    # First try to get transcript in specified language if provided
    try:
        if language_code:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language_code])
        else:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        text = " ".join([t["text"] for t in transcript])
        chunks = text_splitter.split_text(text)
        
        st.info(f"Transcript found! Split into {len(chunks)} chunks.")
        
        # Use map-reduce for longer transcripts
        docs = [Document(page_content=chunk) for chunk in chunks]
        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=True
        )
        return chain.run(docs)
        
    except Exception as e:
        st.warning(f"Could not get transcript directly: {str(e)}")
        
        # Try to list available languages
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            available_languages = [
                f"{t.language_code} ({t.language})" 
                for t in transcript_list
            ]
            st.info(f"Available transcript languages: {', '.join(available_languages)}")
        except:
            st.warning("Could not retrieve available languages.")
        
        # Fallback to YoutubeLoader
        st.info("Attempting to load with YoutubeLoader...")
        loader = YoutubeLoader.from_youtube_url(
            f"https://www.youtube.com/watch?v={video_id}",
            add_video_info=True
        )
        docs = loader.load()
        
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=stuff_prompt)
        return chain.run(docs)

# Function to process website content
def process_website(url, llm):
    st.info("Processing website content...")
    loader = UnstructuredURLLoader(
        urls=[url],
        ssl_verify=False,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
        }
    )
    docs = loader.load()
    
    st.info(f"Content loaded! Document count: {len(docs)}")
    
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=stuff_prompt)
    return chain.run(docs)

# Main execution
if st.button("Summarize the Content from YT or Website"):
    
    if not groq_api_key.strip() and not os.getenv("GROQ_API_KEY"):
        st.error("Please provide a Groq API Key in the sidebar or set it in your environment variables.")
    elif not generic_url.strip():
        st.error("Please provide a URL to summarize.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL (YouTube video or website).")
    else:
        try:
            with st.spinner("Processing... This may take a minute."):
                
                llm = initialize_llm(groq_api_key)
                
                # Check if it's a YouTube URL
                video_id = extract_youtube_id(generic_url)
                
                if video_id:
                    summary = process_youtube(video_id, llm, selected_language)
                else:
                    summary = process_website(generic_url, llm)
                
                
                st.success("Summary generated successfully!")
                st.write("### Summary")
                st.write(summary)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)