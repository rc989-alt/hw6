"""
INFO 4940/5940 Tutor Chatbot
A RAG-powered tutoring assistant for the Applied Machine Learning course
"""

import os
from pathlib import Path
from shiny import App, ui
from chatlas import ChatOpenAI

# Load knowledge base files
def load_knowledge_base():
    """Load all knowledge base documents into a single context string"""
    knowledge_dir = Path("knowledge")
    knowledge_content = []

    knowledge_files = [
        "course-overview.md",
        "hw-06-instructions.md",
        "coding-guidance.md"
    ]

    for filename in knowledge_files:
        filepath = knowledge_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                knowledge_content.append(f"## {filename}\n\n{content}\n\n")

    return "\n".join(knowledge_content)

# Load knowledge base at startup
KNOWLEDGE_BASE = load_knowledge_base()

# System prompt for the tutor
SYSTEM_PROMPT = f"""You are an expert teaching assistant for INFO 4940/5940: Applied Machine Learning at Cornell University.

Your role is to help students with:
- Understanding course concepts, assignments, and projects
- Answering questions about course policies and requirements
- Providing coding guidance in Python or R
- Explaining machine learning concepts and techniques
- Offering study tips and best practices
- Debugging code and troubleshooting issues

Guidelines:
1. Be helpful, encouraging, and patient
2. Provide clear explanations with examples when appropriate
3. Guide students to learn rather than just giving answers
4. Reference specific course materials when relevant
5. Suggest additional resources when helpful
6. For coding questions, provide clear, well-commented code examples
7. If you don't know something, be honest and suggest where to find the answer
8. Encourage good practices: version control, reproducibility, documentation

What you should NOT do:
- Do not complete assignments for students
- Do not provide complete solutions without explanation
- Do not make up information about course policies
- Do not encourage academic dishonesty

You have access to the following course materials:

{KNOWLEDGE_BASE}

Use this information to provide accurate, helpful responses grounded in the actual course content.
"""

# Create the Shiny UI
app_ui = ui.page_fluid(
    ui.head_content(
        ui.tags.style("""
            .app-title {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .app-title h2 {
                margin: 0;
                font-weight: 600;
            }
            .app-title p {
                margin: 5px 0 0 0;
                opacity: 0.9;
            }
            .info-box {
                background-color: #f0f4f8;
                border-left: 4px solid #667eea;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 5px;
            }
            .chat-container {
                max-width: 900px;
                margin: 0 auto;
            }
        """)
    ),
    ui.div(
        {"class": "chat-container"},
        ui.div(
            {"class": "app-title"},
            ui.h2("🎓 INFO 4940/5940 Tutor Assistant"),
            ui.p("Your AI teaching assistant for Applied Machine Learning")
        ),
        ui.div(
            {"class": "info-box"},
            ui.markdown("""
            **Welcome!** I'm here to help you with:
            - 📚 Course concepts and assignments
            - 💻 Python and R coding guidance
            - 📊 Machine learning techniques and best practices
            - 🤔 Questions about course policies and requirements

            Feel free to ask me anything about the course!
            """)
        ),
        ui.chat_ui("chat")
    )
)

def server(input, output, session):
    """Server function with chat integration"""

    # Initialize the OpenAI chat client with system prompt
    chat_client = ChatOpenAI(
        model="gpt-4.1",
        system_prompt=SYSTEM_PROMPT,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create chat instance
    chat = ui.Chat(id="chat")

    # Handle user messages with streaming responses
    @chat.on_user_submit
    async def handle_user_input(user_input: str):
        # Stream the response from the LLM
        response = await chat_client.stream_async(user_input)

        # Append the streaming response to the chat
        await chat.append_message_stream(response)

# Create the Shiny app
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()
