import reflex as rx
from components.ui_components import navbar
from states.rag_state import ChatState

def chat_message(msg: dict) -> rx.Component:
    return rx.box(
        rx.text(msg["content"], color="white"),
        rx.cond(msg["role"] == "assistant", rx.text(f"Sources: {msg['sources']}", font_size="10px", color="gray", margin_top="2")),
        padding="15px", border_radius="lg",
        bg=rx.cond(msg["role"] == "user", "#2b6cb0", "#2d3748"),
        align_self=rx.cond(msg["role"] == "user", "flex-end", "flex-start"),
        max_width="75%"
    )

def chat_page() -> rx.Component:
    return rx.box(
        navbar(),
        rx.center(
            rx.vstack(
                rx.box(
                    rx.vstack(
                        rx.foreach(ChatState.history, chat_message),
                        rx.cond(ChatState.is_typing, rx.text("DocuMind AI is thinking...", color="gray", font_style="italic")),
                        spacing="4", width="100%"
                    ),
                    height="65vh", width="100%", overflow_y="auto", padding="20px", border="1px solid #2a2a3a", border_radius="md", bg="#1a1a2e"
                ),
                rx.hstack(
                    rx.input(placeholder="Ask a question...", value=ChatState.question, on_change=ChatState.set_question, width="100%", bg="#2d3748", color="white", border="none"),
                    rx.button("Send", on_click=ChatState.ask, bg="#3182ce", color="white"),
                    width="100%", padding_top="4"
                ), width="60%", height="80vh", padding_top="5vh"
            )
        ), bg="#0f0f1a", min_height="100vh"
    )
