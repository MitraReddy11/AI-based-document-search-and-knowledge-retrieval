import reflex as rx
from components.ui_components import navbar, footer
from states.rag_state import ChatState

def history_page() -> rx.Component:
    return rx.box(
        navbar(),
        rx.center(
            rx.vstack(
                rx.heading("Session History", color="white"),
                rx.box(
                    rx.foreach(ChatState.history, lambda msg: rx.box(
                        rx.text(rx.cond(msg["role"] == "user", "You:", "AI:"), weight="bold", color="#3182ce"),
                        rx.text(msg["content"], color="white"),
                        padding="10px", border_bottom="1px solid #2a2a3a"
                    )), width="100%", bg="#1a1a2e", border_radius="md", padding="20px"
                ), width="60%", padding_top="5vh", spacing="4"
            ), min_height="80vh"
        ), footer(), bg="#0f0f1a", min_height="100vh"
    )
