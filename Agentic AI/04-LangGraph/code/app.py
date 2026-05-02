"""
app.py — Gradio Application for the Sidekick Agent

This is the entry point for the Sidekick application.
It creates a web UI where users can interact with their personal AI co-worker.

Key Gradio concepts used:
- gr.State: per-session state (each user gets their own Sidekick instance)
- delete_callback: cleanup hook when session ends (closes browser)
- ui.load: called when a new user opens the page (initializes Sidekick)
- button.click: callback when user clicks Go (runs a super step)

Run with: gradio app.py
"""

import gradio as gr
from sidekick import Sidekick


async def setup():
    """
    Called when a new user loads the UI.
    Creates a fresh Sidekick instance with its own browser, tools, and graph.
    Each user session gets an independent Sidekick.
    """
    sidekick = Sidekick()
    await sidekick.setup()  # Async: launches browser, builds graph
    return sidekick


async def process_message(sidekick, message, success_criteria, history):
    """
    Called when user clicks Go or presses Enter.
    Runs one super step: the full graph executes (worker ⇄ tools ⇄ evaluator)
    and returns the result to display in the chat.
    """
    results = await sidekick.run_superstep(message, success_criteria, history)
    return results, sidekick


async def reset():
    """
    Called when user clicks Reset.
    Creates a completely fresh Sidekick (new browser, new graph, new memory).
    """
    new_sidekick = Sidekick()
    await new_sidekick.setup()
    return "", "", None, new_sidekick


def free_resources(sidekick):
    """
    Called automatically when a user's session ends (closes tab, etc.).
    Cleans up the Playwright browser and resources.
    """
    print("Cleaning up")
    try:
        if sidekick:
            sidekick.cleanup()
    except Exception as e:
        print(f"Exception during cleanup: {e}")


# --- GRADIO UI DEFINITION ---
with gr.Blocks(title="Sidekick", theme=gr.themes.Default(primary_hue="emerald")) as ui:
    gr.Markdown("## Sidekick Personal Co-Worker")

    # gr.State stores per-session data — each user gets their own Sidekick
    # delete_callback ensures browser cleanup when session ends
    sidekick = gr.State(delete_callback=free_resources)

    # Chat display
    with gr.Row():
        chatbot = gr.Chatbot(label="Sidekick", height=300, type="messages")

    # Input fields
    with gr.Group():
        with gr.Row():
            message = gr.Textbox(show_label=False, placeholder="Your request to the Sidekick")
        with gr.Row():
            success_criteria = gr.Textbox(
                show_label=False, placeholder="What are your success criteria?"
            )

    # Action buttons
    with gr.Row():
        reset_button = gr.Button("Reset", variant="stop")
        go_button = gr.Button("Go!", variant="primary")

    # --- LIFECYCLE CALLBACKS ---

    # On page load: create and initialize a Sidekick for this session
    ui.load(setup, [], [sidekick])

    # On Enter in message box: run a super step
    message.submit(
        process_message, [sidekick, message, success_criteria, chatbot], [chatbot, sidekick]
    )

    # On Enter in success criteria box: run a super step
    success_criteria.submit(
        process_message, [sidekick, message, success_criteria, chatbot], [chatbot, sidekick]
    )

    # On Go button click: run a super step
    go_button.click(
        process_message, [sidekick, message, success_criteria, chatbot], [chatbot, sidekick]
    )

    # On Reset button click: create fresh Sidekick
    reset_button.click(reset, [], [message, success_criteria, chatbot, sidekick])


# Launch the app (opens browser automatically)
ui.launch(inbrowser=True)
