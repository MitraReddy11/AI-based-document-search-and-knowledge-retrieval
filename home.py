import reflex as rx
from components.ui_components import navbar, footer, feature_card

def home_page() -> rx.Component:
    return rx.box(
        navbar(),
        rx.center(
            rx.vstack(
                rx.heading("DocuMind AI", size="9", color="white", weight="bold"),
                rx.text("Transform Documents into Knowledge using Corrective RAG.", color="#a0aec0", size="4"),
                rx.hstack(
                    rx.button("Upload Docs", on_click=rx.redirect("/upload"), bg="#3182ce", color="white", size="3"),
                    rx.button("Start Chat", on_click=rx.redirect("/chat"), variant="outline", color_scheme="blue", size="3"),
                    spacing="4", margin_top="4"
                ),
                rx.hstack(
                    feature_card("Smart Retrieval", "Uses ChromaDB to fetch relevant text chunks instantly."),
                    feature_card("Corrective RAG", "LLM checks relevance and rewrites queries for better accuracy."),
                    feature_card("Fast Responses", "Powered by Groq's blazing-fast LLaMA 3 API."),
                    spacing="5", margin_top="8"
                ),
                align_items="center", spacing="4"
            ), height="80vh"
        ),
        footer(),
        background="radial-gradient(circle at 50% 0%, #16213e 0%, #0f0f1a 100%)", min_height="100vh"
    )
