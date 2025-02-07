"""Building the retriever"""
import uuid
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.documents import Document
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from qdrant_client.models import Distance, VectorParams


class Retriever:
    def __init__(self, llm, embedding_model):
        super().__init__()
        self.llm =llm
        self.embedding_model = embedding_model


    def summarize_text_tables(self, texts, tables):
        text_table_summarization_prompt = """
        You are an assistant responsible for summarizing the provided {element_type} from the "Attention is all you need" paper. 
        Your task is to generate a concise summary of the content found in the chunk below.

        Instructions:
        - Respond only with the summary.
        - Do not include any additional commentary, greetings, or extra text.
        - Do not begin your response with phrases like "Here is a summary" or similar introductions.
        - Simply output the summary as it is.

        Chunk: {element}
        """
        prompt = PromptTemplate.from_template(text_table_summarization_prompt)
        summarization_chain = prompt | self.llm | StrOutputParser()

        # summarize texts
        text_inputs = [{"element_type": "text", "element": text} for text in texts]
        text_summaries = summarization_chain.batch(text_inputs)

        # summarize tables
        tables_inputs = [{"element_type": "table", "element": table} for table in tables]
        tables_summaries = summarization_chain.batch(tables_inputs)

        return text_summaries, tables_summaries


    def summrize_images(self, images):
        image_summarization_prompt = """
        You are an expert in analyzing images.
        Your task is to provide a detailed, technical breakdown of the given image. 

        - If the image contains a graph, describe the axes, labels, and key trends. Identify whether it is a bar chart, line plot, or scatter plot, and explain what it represents.
        - If the image contains a diagram, break it down into key components, explaining their roles in the Transformer architecture.
        - If the image contains mathematical notation, provide an interpretation of the equations or symbols.
        - If the image represents text, summarize its content concisely.
        """
        messages = [
            (
                "user", 
                [
                    {"type": "text", "text": image_summarization_prompt},
                    {"type": "image_url", "image_url":
                        {"url": "data:image/jpeg;base64,{image_data}"}}
                ]
            )
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        summarization_chain = prompt | self.llm | StrOutputParser()
        image_summaries = summarization_chain.batch(images)
        return image_summaries


    def add_summary_to_retriever(self, elements, summaries, retriever, id_key):
        doc_ids = [str(uuid.uuid4()) for _ in elements]
        summary_docs = [
             Document(page_content=summary, metadata={id_key: doc_ids[i]})
            for i, summary in enumerate(summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, elements)))


    def make_retriever(self, texts, tables, images):
        qdrant_client = QdrantClient(":memory:")
        collection_name = "multi_rag"
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
        vectorstore = Qdrant(
            client=qdrant_client,
            collection_name=collection_name,
            embeddings=self.embedding_model
        )
        store = InMemoryStore()
        id_key = "doc_id"

        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key=id_key,
        )

        # getting the text, tables and images summaries and add them to the retriever
        texts_summary, tables_summary = self.summarize_text_tables(texts, tables)
        images_summary = self.summrize_images(images)

        self.add_summary_to_retriever(texts, texts_summary, retriever, id_key)
        self.add_summary_to_retriever(tables, tables_summary, retriever, id_key)
        self.add_summary_to_retriever(images, images_summary, retriever, id_key)

        return retriever
