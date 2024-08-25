# run with chainlit run chatbot.py -w --port 8000
import chainlit as cl
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain_core.prompts.chat import ChatPromptTemplate
from dotenv import dotenv_values
from langchain.vectorstores import pgvector
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage,trim_messages

ollama_host="http://ollama:11434"
embedmodel="mxbai-embed-large"
def connectStr(alchemy=False): 
    if alchemy: return f'postgresql+psycopg://postgres:{dotenv_values()["POSTGRES_PASSWORD"]}@pgvector:5432/ragtest'
    return  f'host=pgvector dbname=ragtest user=postgres password={dotenv_values()["POSTGRES_PASSWORD"]}'

def getVectorStore():
    embeddings = OllamaEmbeddings(base_url=ollama_host,model=embedmodel)
    return pgvector.PGVector(connectStr(True),embeddings)    

@cl.on_chat_start
def quey_llm():
    llm = Ollama(model='llama3',base_url=ollama_host,temperature=0)

    general_system_template = r""" 
    Given a specific context, please give a concise but exhaustive answer to the question.
    If you don't know the answer, just say that you don't know.
    Always report any link you find
    ----
    {context}
    ----
    """
    contextualized_templ=r""" 
    Given a chat history and the latest user question, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",general_system_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"), ] )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualized_templ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"), ] )

    history_aware_retriever = create_history_aware_retriever(
        llm, getVectorStore().as_retriever(), contextualize_q_prompt
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    cl.user_session.set("llm_chain", rag_chain)



@cl.on_message
async def query_llm(message: cl.Message):
    llm_chain = cl.user_session.get("llm_chain")
    chat_history = cl.user_session.get("history")
    if chat_history ==None : chat_history = []
    if llm_chain!=None:
        msg = cl.Message(content="")
        baseurl = dotenv_values()["JIRASDBASEURL"]
        elements = []
        async for chunk in llm_chain.astream(
            {"input": message.content,"chat_history": chat_history},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            if "answer" in chunk : 
                await msg.stream_token(chunk["answer"])
            if "context" in chunk:
                for d in chunk["context"]:
                    meta = d.metadata
                    url=f'{baseurl}/servicedesk/customer/portal/6/article/{meta["source"]}'
                    elements.append(cl.Text(content=f'[{meta["title"]}]({url})',display="inline"))
        msg.elements=elements
        await msg.send()
        chat_history.extend([HumanMessage(message.content),AIMessage(msg.content)])
        trimmer = trim_messages( max_tokens=65, strategy="last", token_counter=llm_chain, include_system=True, allow_partial=False, start_on="human", )
        trimmer.invoke(chat_history)
        cl.user_session.set("history",chat_history)
