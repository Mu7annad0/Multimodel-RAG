"""Generator"""
import re
import base64
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.ingestion import Parser
from src.retriever import Retriever
from src.evaluation import RAGEvaluator


class Generator:
    """Generate responses from multimodal PDF documents using LLM and embedding models."""

    def __init__(self, pdf_path, llm, embedding_model):
        super().__init__()
        self.llm = llm
        self.embedding_model = embedding_model
        self.parser = Parser(pdf_path)
        self.retriever = Retriever(llm, embedding_model)
        self.evaluater = RAGEvaluator(llm)


    def parse(self):
        """Parse the PDF and return its processed content."""
        return self.parser.process()


    def looks_like_base64(self, sb):
        """Check if the string looks like base64"""
        return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None

    def is_image_data(self, b64data):
        """
        Check if the base64 data is an image by looking at the start of the data
        """
        image_signatures = {
            b"\xff\xd8\xff": "jpg",
            b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
            b"\x47\x49\x46\x38": "gif",
            b"\x52\x49\x46\x46": "webp",
        }
        try:
            header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
            for sig, format in image_signatures.items():
                if header.startswith(sig):
                    return True
            return False
        except Exception:
            return False


    def parse_documents(self, docs):
        """Separate the documents into images and texts based on base64 validation."""
        b64 = []
        text = []
        for doc in docs:
            if self.looks_like_base64(doc) and self.is_image_data(doc):
                b64.append(doc)
            else:
                text.append(doc)
        return {"images": b64, "texts": text}


    def building_prompt(self, settings):
        """Construct and return a ChatPromptTemplate using context texts, images, and a question from settings."""
        docs = settings["context"]
        question = settings["question"]

        context_text = ""
        if len(docs["texts"]) > 0:
            for text_element in docs["texts"]:
                if isinstance(text_element, list):
                    # Join list items with a space
                    context_text += " ".join(text_element)
                else:
                    context_text += text_element

        generation_prompt = f"""
        You are an AI engineering expert analyzing multimodal documents. 
        Using the information contained in the context—which may include text, tables, or images—provide a comprehensive and seamless answer to the question.
        
        Mathematical expressions should be formatted using LaTeX-style notation in Markdown. 
        - Inline formulas must be enclosed in single dollar signs: `$...$`
        - Block formulas must be enclosed in double dollar signs: `$$...$$`

        Before answering, carefully analyze the question and identify all relevant details in the context, including technical concepts, numerical data, and visual patterns. 
        Ensure that all formulas and technical elements are presented using LaTeX notation correctly.

        Your answer should integrate insights from the context naturally without using explicit section titles or headings. 
        Use Markdown formatting where appropriate, especially for highlighting numerical data, and maintain a clear, concise, and objective tone.

        If the context includes images, assume they will be plotted, so you can reference them in your answer where relevant.

        If the answer cannot be deduced from the provided context, request clarification on the specific missing elements rather than speculating.

        Context:
        {context_text}

        Question:
        {question}
        """

        prompt_content = [{"type": "text", "text": generation_prompt}]
        if len(docs["images"]) > 0:
            for image in docs["images"]:
                prompt_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    }
                )

        return ChatPromptTemplate.from_messages(
            [
                HumanMessage(content=prompt_content)
            ]
        )


    def create_chain(self):
        """Build and return a processing chain that retrieves context and generates a response using the LLM."""
        texts, tables, images = self.parse()
        texts, tables, images = self.parse()
        retriever = self.retriever.make_retriever(texts, tables, images)

        chain = {
            "context": retriever | RunnableLambda(self.parse_documents),
            "question": RunnablePassthrough()
        } | RunnablePassthrough().assign(
            response = (
                RunnableLambda(self.building_prompt)
                | self.llm
                | StrOutputParser()
            )
        )
        return chain
