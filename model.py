from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS, Pinecone
from langchain.llms import CTransformers, OpenAI
from langchain.chains import (
    RetrievalQA,
    OpenAPIEndpointChain,
    HypotheticalDocumentEmbedder,
)
import os
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI

import chainlit as cl

os.environ["OPENAI_API_KEY"] = "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"

DB_FAISS_PATH = "vectorstores/db_faiss"

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, please just say that you don't know the answer, don't try to make up an answer.


Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer: 
"""

mike_prompt_template = """Give me the best response to the following question. Even if you don't know the answer.
{question}

Answer:
"""


def set_custom_prompt():
    prompt = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )
    return prompt


def set_mike_prompt():
    prompt = PromptTemplate(template=mike_prompt_template, input_variables=["question"])
    return prompt


def load_llm():
    # llm = CTransformers(
    #     model="llama-2-7b-chat.ggmlv3.q8_0.bin",
    #     model_type="llama",
    #     max_new_tokens=512,
    #     temerature=0.5,
    # )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    return llm


def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 6}),
        chain_type_kwargs={"prompt": prompt},
    )

    # OpenAPIEndpointChain()

    return qa_chain


def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    # embeddings = OpenAIEmbeddings(
    #     openai_api_key="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",
    #     model="text-embedding-ada-002",
    #     deployment="tj-and-mike",
    # )

    llm = load_llm()
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    # prompt = set_custom_prompt()
    # prompt = set_mike_prompt()

    # llm = LLMChain(llm=OpenAI(), prompt=prompt)

    # llm_chain = LLMChain(llm=llm, prompt=prompt)

    # embeddings = HypotheticalDocumentEmbedder.from_llm(
    #     llm=llm, base_embeddings=base_embeddings, prompt_key=="web_search"
    # )

    # pinecone.init(
    #     api_key="f2c105ce-da2c-45d7-b1fa-830d455e06a0", environment="us-west1-gcp-free"
    # )
    # index = pinecnoe.Index("chainlit")

    # vectorstore = Pinecone(index, embeddings.embed_query, "text")

    db = FAISS.load_local(DB_FAISS_PATH, embeddings)

    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa


def final_result(query):
    qa_result = qa_bot()
    response = qa_result({"query": query})
    return response


@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to the SwiftUI Bot. What is your question?"
    await msg.update()
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    print(res)
    answer = res["result"]

    await cl.Message(content=answer).send()
