import reflex as rx
from components.ui_components import navbar, footer
from states.rag_state import ChatState

def upload_page() -> rx.Component:
    return rx.box(
        navbar(),
        rx.center(
            rx.vstack(
                rx.heading("Upload Knowledge Base", color="white"),
                rx.text("Supported files: PDF, TXT", color="gray"),
                
                rx.upload(
                    rx.vstack(
                        rx.icon("upload", size=32, color="gray"),
                        rx.text("Drag & Drop or Click to Select (PDF / TXT)", color="white"),
                        align="center"
                    ),
                    id="doc_upload",
                    multiple=True,
                    padding="50px",
                    border="2px dashed #4a5568",
                    border_radius="md",
                    _hover={"bg": "#2d3748"}
                ),
                
                rx.cond(
                    rx.selected_files("doc_upload"),
                    rx.text("Files ready to process. Click below!", color="yellow")
                ),
                
                # The Upload Button
                rx.button(
                    "Process & Upload Documents",
                    on_click=lambda: ChatState.handle_upload(rx.upload_files(upload_id="doc_upload")),
                    bg="#3182ce",
                    color="white",
                    size="3",
                    margin_top="4"
                ),
                
                rx.heading("Uploaded Files:", size="4", color="white", margin_top="4"),
                rx.foreach(
                    ChatState.uploaded_files,
                    lambda filename: rx.text("✅ " + filename, color="green")
                ),
                
                width="50%",
                align_items="center",
                spacing="4"
            ),
            height="80vh"
        ),
        footer(),
        bg="#0f0f1a",
        min_height="100vh"
    )
