# GRADIO USER INTERFACE - Agent System Frontend Integration
# Demonstrates streaming UI updates and async callback patterns
# Key concepts: Generator-based streaming, event handling, and user experience
#
# HOW TO RUN:
# 1. Open terminal (Ctrl+` or View > Terminal)
# 2. cd 2_openai/deep_research
# 3. uv run deep_research.py

import gradio as gr
from dotenv import load_dotenv
from research_manager import ResearchManager

load_dotenv(override=True)

# Async callback function for Gradio - enables streaming updates
async def run(query: str):
    """Streams research progress updates to UI in real-time"""
    async for chunk in ResearchManager().run(query):
        yield chunk  # Generator pattern enables incremental UI updates

# Gradio UI definition with custom theme and event handling
with gr.Blocks(theme=gr.themes.Default(primary_hue="sky")) as ui:
    gr.Markdown("# Deep Research")
    query_textbox = gr.Textbox(label="What topic would you like to research?")
    run_button = gr.Button("Run", variant="primary")
    report = gr.Markdown(label="Report")
    
    # Event binding: Button click → Async callback → Streaming output
    run_button.click(fn=run, inputs=query_textbox, outputs=report)
    # Alternative trigger: Enter key submission
    query_textbox.submit(fn=run, inputs=query_textbox, outputs=report)

# Launch with browser auto-open for immediate user access
ui.launch(inbrowser=True)

