import reflex as rx
from components.ui_components import navbar, footer

def about_page() -> rx.Component:
    return rx.box(
        navbar(),
        rx.center(
            rx.vstack(
                rx.heading("About DocuMind AI", color="white"),
                rx.text("This application demonstrates a full-stack GenAI application using Corrective RAG.", color="gray"),
                width="60%", padding_top="5vh", spacing="4"
            ), min_height="80vh"
        ), footer(), bg="#0f0f1a", min_height="100vh"
    )
