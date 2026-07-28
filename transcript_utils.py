import re
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_video_id(url):
    # Extracts the 11-character YouTube video ID from a standard video URL
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None


def get_transcript(url):
    # Fetches the English transcript for a video, preferring a manual one over an auto-generated one
    video_id = get_video_id(url)
    if not video_id:
        return None

    ytt_api = YouTubeTranscriptApi()
    transcripts = ytt_api.list(video_id)

    transcript = ""
    for t in transcripts:
        if t.language_code == 'en':
            if t.is_generated:
                if len(transcript) == 0:
                    transcript = t.fetch()
            else:
                transcript = t.fetch()
                break

    return transcript if transcript else None


def process(transcript):
    # Converts the raw transcript object into a single "Text ... Start ..." string
    txt = ""
    for i in transcript:
        try:
            txt += f"Text: {i.text} Start: {i.start}\n"
        except KeyError:
            pass
    return txt


def chunk_transcript(processed_transcript, chunk_size=200, chunk_overlap=20):
    # Splits the processed transcript into overlapping chunks for embedding/search
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(processed_transcript)
