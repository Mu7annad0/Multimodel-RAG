"""Streanlit demo"""
import io
import base64
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.generation import Generator
load_dotenv()


# Define a class that encapsulates your chain.
class YourChainClass:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.embedding_model = OpenAIEmbeddings()
        
        # Specify your PDF file path.
        pdf_path = "/Users/mu7annad.0gmail.com/Documents/GitHub/Multimodel-RAG/Data/Attention-is-all-you-need.pdf"
        
        generator = Generator(pdf_path, self.llm, self.embedding_model)
        
        # Build your chain.
        self.chain = generator.create_chain()
    
    def invoke(self, prompt):
        """Invoke the chain with the given prompt and return the response."""
        return self.chain.invoke(prompt)


def initialize_chat():
    """Initialize the chat session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chain" not in st.session_state:
        st.session_state.chain = YourChainClass()


def display_images_grid(images, target_height=300):
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

            # Display the image in the appropriate column
            with grid[idx % cols]:
                st.image(img, caption=f"Image {idx + 1}")

        except Exception as e:
            with grid[idx % cols]:
                st.error(f"Error displaying image {idx + 1}: {str(e)}")


def main():
    st.title("PDF Chat Interface")
    initialize_chat()

    # Display chat history (text and images)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("images"):
                display_images_grid(message["images"])

    if prompt := st.chat_input("Ask a question about the PDF"):
        # Save and display the user message.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chain.invoke(prompt)

                # Display the text portion of the response.
                st.markdown(response["response"])

                new_message = {"role": "assistant", "content": response["response"]}

                # If there are images in the response, display them and store them.
                if "context" in response and "images" in response["context"]:
                    images = response["context"]["images"]
                    display_images_grid(images)
                    new_message["images"] = images

                st.session_state.messages.append(new_message)


if __name__ == "__main__":
    main()
