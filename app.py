import io
import base64
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.generation import Generator
from typing import Dict, List, Any

load_dotenv()

class ChainClass:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)  # Enable streaming
        self.embedding_model = OpenAIEmbeddings()
        # Specify your PDF file path
        pdf_path = "Data/Attention-is-all-you-need.pdf"
        generator = Generator(pdf_path, self.llm, self.embedding_model)
        self.chain = generator.create_chain()

    async def stream(self, prompt: str) -> Dict[str, Any]:
        """Stream the response from the chain with the given prompt."""
        response_placeholder = st.empty()
        collected_chunks = []
        collected_messages = []
        
        # Initialize the response dictionary
        final_response = {
            "response": "",
            "context": {"images": []}
        }
        
        async for chunk in self.chain.astream(prompt):
            if isinstance(chunk, dict):
                if "response" in chunk:
                    collected_chunks.append(chunk["response"])
                    collected_messages.append(chunk["response"])
                    text = "".join(collected_messages)
                    # Update the placeholder with the latest text
                    response_placeholder.markdown(text + "▌")
                    final_response["response"] = text
                
                if "context" in chunk and "images" in chunk["context"]:
                    final_response["context"]["images"] = chunk["context"]["images"]
        
        # Clear the placeholder and return the final response
        response_placeholder.empty()
        return final_response

def initialize_chat():
    """Initialize the chat session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chain" not in st.session_state:
        st.session_state.chain = ChainClass()

def display_images_grid(images: List[str], target_height: int = 300):
    """Display Base64 images in a grid layout with equal heights."""
    if not images:
        return
    
    n_images = len(images)
    cols = min(n_images, 2)
    grid = st.columns(cols)
    
    for idx, img_base64 in enumerate(images):
        try:
            img_bytes = base64.b64decode(img_base64)
            img = Image.open(io.BytesIO(img_bytes))
            
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            
            aspect_ratio = img.size[0] / img.size[1]
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            with grid[idx % cols]:
                st.image(img, caption=f"Image {idx + 1}")
        except Exception as e:
            with grid[idx % cols]:
                st.error(f"Error displaying image {idx + 1}: {str(e)}")

async def main():
    st.title("PDF Chat Interface")
    initialize_chat()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("images"):
                display_images_grid(message["images"])

    if prompt := st.chat_input("Ask a question about the PDF"):
        # Save and display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response with streaming
        with st.chat_message("assistant"):
            response = await st.session_state.chain.stream(prompt)
            
            # Display the final text response
            st.markdown(response["response"])
            
            new_message = {
                "role": "assistant",
                "content": response["response"]
            }

            # If there are images, display and store them
            if "context" in response and "images" in response["context"]:
                images = response["context"]["images"]
                display_images_grid(images)
                new_message["images"] = images

            st.session_state.messages.append(new_message)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())