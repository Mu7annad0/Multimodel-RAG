"""Evaluation"""
import re
import os
from src.generation import Generator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class RAGEvaluator:
    """Evaluates generated text using RAG metrics."""
    def __init__(self, llm):
        super().__init__()
        self.llm = llm


    def evaluate(self, chain, questions, reference_answers):
        """Runs all evaluation metrics and returns combined results"""
        assert len(questions) == len(reference_answers), "Number of questions and reference answers should be the same"

        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(script_dir, "log")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "log.txt")

        with open(log_file, "a") as f:

            f.write("====================================================\n")
            scores = []
            for i, question in enumerate(questions):
                score=0
                print(f"==========Evaluate question No.{i+1}==========")
                response = chain.invoke(question)
                f.write(f"Question: {response['question']}\n")
                f.write(f"Response: {response['response']}\n\n")

                # Retrieval Evaluation (Context Quality)
                score = (self.question_groundedness(response, f)) * 0.5
                score += (self.context_relevance(response, f)) * 0.5

                # Generation Evaluation (Answer Quality)
                score += (self.answer_faithfulness(response, f)) * 0.5
                score += (self.answer_relevance(response, f)) * 0.5

                # Reference-Based Evaluation (Absolute Quality)
                score += self.response_reference_similarity(response, reference_answers[i], f)
                f.write("--------------\n\n")
                scores.append(score)

            final_score = round((sum(scores) / (15*len(questions))) * 100, 2)
            f.write(f"Final score: {final_score}\n")
            f.write("====================================================\n\n\n")


    def question_groundedness(self, response, f):
        """Evaluates if the question can be answered using the given context"""

        prompt = """
        You will be given a context and a question.
        Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
        Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

        Provide your answer as follows:

        Answer:::
        Evaluation: (your rationale for the rating, as a text)
        Total rating: (your rating, as a number between 1 and 5)

        You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

        Now here are the question and context.

        Question: {question}
        Context: {context}
        Answer::: """

        result = (self.llm.invoke(prompt.format(
            context=response["context"],
            question=response["question"],
        ))).content

        score = self.extract_evaluation_score(result)
        f.write(f"question_groundedness score: {score}\n")
        return score


    def context_relevance(self, response, f):
        """Evaluates how relevant the retrieved context is to the question"""

        prompt = """
        Evaluate how relevant the retrieved context is to the given question.
        Rate on a scale of 1-5 where:
        1: Completely irrelevant
        2: Mostly irrelevant with few useful pieces
        3: Partially relevant
        4: Mostly relevant with minor irrelevant parts
        5: Completely relevant
        
        Answer:::
        Evaluation: (your rationale)
        Total rating: (1-5)
        
        Question: {question}
        Context: {context}
        Answer::: """

        result = self.llm.invoke(prompt.format(
            context=response["context"],
            question=response["question"]
        ))
        score = self.extract_evaluation_score(result.content)
        f.write(f"context_relevance score: {score}\n")
        return score


    def answer_faithfulness(self, response, f):
        """Evaluates if the answer is faithful to the given context"""

        prompt = """
        Evaluate if the answer is faithful to and supported by the given context.
        Rate on a scale of 1-5 where:
        1: Answer contradicts context
        2: Answer mostly unsupported by context
        3: Answer partially supported by context
        4: Answer mostly supported with minor unsupported details
        5: Answer completely supported by context

        Answer:::
        Evaluation: (your rationale)
        Total rating: (1-5)
            
        Context: {context}
        Question: {question}
        Answer: {answer}
        Answer::: """

        result = self.llm.invoke(prompt.format(
            context=response["context"],
            question=response["question"],
            answer=response["response"]
        ))
        score = self.extract_evaluation_score(result.content)
        f.write(f"answer_faithfulness score: {score}\n")
        return score


    def answer_relevance(self, response, f):
        """Evaluates how well the answer addresses the question"""

        prompt = """
        Evaluate how well the answer addresses the question.
        Rate on a scale of 1-5 where:
        1: Answer doesn't address the question
        2: Answer barely addresses the question
        3: Answer partially addresses the question
        4: Answer mostly addresses the question
        5: Answer fully addresses the question
        
        Answer:::
        Evaluation: (your rationale)
        Total rating: (1-5)
        
        Question: {question}
        Answer: {answer}
        Answer::: """

        result = self.llm.invoke(prompt.format(
            question=response["question"],
            answer=response["response"]
        ))
        score = self.extract_evaluation_score(result.content)
        f.write(f"answer_relevance score: {score}\n")
        return score


    def response_reference_similarity(self, response, reference_response, f):
        """How close is the response to ideal answer?"""

        prompt = """
        ###Task Description:
        An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
        1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
        2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.

        Answer:::
        Evaluation: (your rationale for the rating, as a text in one line maximum)
        Total rating: (your rating, as a number between 1 and 5)

        Please do not generate any other opening, closing, and explanations. Be sure to include 'Total rating' in your output.
        ###The instruction to evaluate:
        {instruction}

        ###Response to evaluate:
        {response}

        ###Reference Answer (Score 5):
        {reference_answer}

        ###Score Rubrics:
        [Is the response correct, accurate, and factual based on the reference answer?]
        Score 1: The response is completely incorrect, inaccurate, and/or not factual.
        Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
        Score 3: The response is somewhat correct, accurate, and/or factual.
        Score 4: The response is mostly correct, accurate, and factual.
        Score 5: The response is completely correct, accurate, and factual.

        ###Evaluation:"""

        result = (self.llm.invoke(prompt.format(
            instruction=response["question"],
            response=response["response"],
            reference_answer=reference_response
        )
        )).content

        score = self.extract_evaluation_score(result)
        f.write(f"response_reference_similarity score: {score}\n")
        return score


    def extract_evaluation_score(self, text):
        """Extracts the evaluation score from the text."""
        eval_match = re.search(r"Evaluation:\s*(.*?)\s*Total rating:", text, re.DOTALL)
        evaluation_text = eval_match.group(1).strip() if eval_match else None

        # Extract the total rating (digits following "Total rating:")
        rating_match = re.search(r"Total rating:\s*(\d+)", text)
        total_rating = rating_match.group(1) if rating_match else None

        return int(total_rating)



def main():
    pdf = "/Users/mu7annad.0gmail.com/Documents/GitHub/Multimodel-RAG/Data/Attention-is-all-you-need.pdf"
    llm = llm = ChatOpenAI(model="gpt-4o-mini")
    embedding_model = OpenAIEmbeddings()

    # List of questions
    questions = [
        "How is the scaled dot product attention calculated?",
        "What is the BLEU score of the model in English to German translation (EN-DE)?",
        "How long were the base and big models trained?",
        "Which optimizer was used when training the models?",
        "What is the position-wise feed-forward neural network mentioned in the paper?"
    ]

    # List of answers based only on 'Attention Is All You Need'
    answers = [
        "The scaled dot-product attention takes queries (Q), keys (K), and values (V). It computes attention scores by performing the dot product between Q and K, scaling by the square root of the key dimension sqrt(d_k), and applying a softmax function. The final output is obtained by weighting the values (V) using these attention scores: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.",
        
        "The Transformer achieved a BLEU score of 27.3 for the base model and 28.4 for the big model on the WMT 2014 English-to-German translation task.",
        
        "The base model was trained for 100,000 steps (approximately 12 hours) on 8 NVIDIA P100 GPUs. The big model was trained for 300,000 steps (approximately 3.5 days) on 8 NVIDIA P100 GPUs.",
        
        "The Adam optimizer was used with the following hyperparameters: β1 = 0.9, β2 = 0.98, ε = 10^-9. The learning rate followed a warm-up schedule: lrate = d_model^-0.5 * min(step_num^-0.5, step_num * warmup_steps^-1.5), where warmup_steps = 4000.",
        
        "The position-wise feed-forward network (FFN) is applied to each position separately and identically across different positions. It consists of two linear transformations with a ReLU activation in between: FFN(x) = max(0, xW1 + b1) W2 + b2. The input and output dimensions are d_model = 512, the hidden layer has a dimension of d_ff = 2048, and it can also be viewed as two convolutions with kernel size 1."
    ]

    gen = Generator(pdf, llm, embedding_model)
    chain = gen.create_chain()

    evaluater = RAGEvaluator(llm)
    evaluater.evaluate(chain, questions, answers)


if __name__ == "__main__":
    # main()
