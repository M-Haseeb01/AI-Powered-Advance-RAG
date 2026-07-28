import gradio as gr
from transcript_utils import get_transcript, process, chunk_transcript
from llm_utils import (
    get_llm,
    get_embedding_model,
    create_faiss_index,
    create_summary_chain,
    create_qa_chain,
    generate_answer,
)

processed_transcript = ""

CUSTOM_CSS = """
.gradio-container {max-width: 900px !important; margin: auto !important;}
#header {text-align: center; margin-bottom: 0.5rem;}
#header h1 {font-size: 2.1rem; margin-bottom: 0.2rem;}
#header p {color: #6b7280; font-size: 1rem;}
#url-row {gap: 0.5rem;}
.section-card {border-radius: 14px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08);}
footer {display: none !important;}
"""

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)


def summarize_video(video_url):
    # Fetches the transcript (if needed) and returns a one-paragraph summary of the video
    global processed_transcript

    if not video_url:
        return "⚠️ Please provide a valid YouTube URL."

    fetched_transcript = get_transcript(video_url)
    processed_transcript = process(fetched_transcript) if fetched_transcript else ""

    if not processed_transcript:
        return "❌ No transcript available for this video."

    llm = get_llm()
    summary_chain = create_summary_chain(llm)
    return summary_chain.run({"transcript": processed_transcript})


def answer_question(video_url, user_question):
    # Fetches the transcript (if needed), retrieves relevant context, and answers the user's question
    global processed_transcript

    if not processed_transcript:
        if not video_url:
            return "⚠️ Please provide a valid YouTube URL."
        fetched_transcript = get_transcript(video_url)
        processed_transcript = process(fetched_transcript) if fetched_transcript else ""

    if not processed_transcript or not user_question:
        return "⚠️ Please provide a valid question and ensure the transcript has been fetched."

    chunks = chunk_transcript(processed_transcript)
    llm = get_llm()
    embedding_model = get_embedding_model()
    faiss_index = create_faiss_index(chunks, embedding_model)
    qa_chain = create_qa_chain(llm)
    return generate_answer(user_question, faiss_index, qa_chain)


with gr.Blocks(theme=THEME, css=CUSTOM_CSS, title="YouTube Summarizer & Q&A") as interface:
    with gr.Column(elem_id="header"):
        gr.Markdown("# 🎬 YouTube Video Summarizer & Q&A")
        gr.Markdown("Paste a YouTube link, get an instant summary, then ask follow-up questions about the video.")

    with gr.Group(elem_classes="section-card"):
        with gr.Row(elem_id="url-row"):
            video_url = gr.Textbox(
                label="YouTube Video URL",
                placeholder="https://www.youtube.com/watch?v=...",
                scale=4,
                show_label=True,
            )

    with gr.Tabs():
        with gr.Tab("📄 Summary"):
            with gr.Group(elem_classes="section-card"):
                summarize_btn = gr.Button("Summarize Video", variant="primary", size="lg")
                summary_output = gr.Textbox(
                    label="Video Summary",
                    lines=6,
                    show_copy_button=True,
                    placeholder="Your summary will appear here...",
                )

        with gr.Tab("💬 Ask a Question"):
            with gr.Group(elem_classes="section-card"):
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g. What are the main takeaways from this video?",
                )
                question_btn = gr.Button("Ask", variant="primary", size="lg")
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=6,
                    show_copy_button=True,
                    placeholder="Your answer will appear here...",
                )

    gr.Markdown(
        "<div style='text-align:center; color:#9ca3af; margin-top:1rem; font-size:0.85rem;'>"
        "Powered by open-source LLMs via Hugging Face Inference API"
        "</div>"
    )

    summarize_btn.click(summarize_video, inputs=video_url, outputs=summary_output)
    question_btn.click(answer_question, inputs=[video_url, question_input], outputs=answer_output)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
