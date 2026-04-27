import uuid

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy.testing.plugin.plugin_base import config

load_dotenv()
import os
from langchain.tools import tool

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class CustomAgent(AgentState):
    user_id: str
    thread_id: str
    query: str

checkpointer = InMemorySaver()
llm = ChatOpenAI(model="gpt-5.2")
prompt_template = ChatPromptTemplate.from_template(
    """You are a RAG Agent, and help user to search for the information to the given query"""
    """You can use the tools to give more detailed answer"""
    """You will return the lastest answer and the most relevant information"""
)

@tool("tavily_search", description = "search detailed information")
def tavily_search(query):
    tavily_tool = TavilySearch(max_results=5, search_depth = "advanced")
    tavily_result = tavily_tool.invoke(query)
    formed_response = "\n\n".join(f"Title: {res.metadata['title']}\n\nContent: {res.metadata['content']}" for res in tavily_result)
    return formed_response


from langchain.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from webloader import file_content


@tool("rag_retriever", description="Rag Retriever")
def rag_retriever(query:str):
    """Retrieve the contents for a given query
    Args:
        query (str): The query to be retrieved
    Returns:
        str: The content of the query

    """
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size= 10000,
        chunk_overlap= 500,
    )
    texts = text_spliter.split_documents(documents=file_content)
    embedding = OpenAIEmbeddings(model="")
    vectorstores = InMemoryVectorStore.from_documents(documents=texts, embedding=embedding)
    retriever = vectorstores.similarity_search(query=query, k=5)
    serialize_return = "\n\n".join(f"Source: {doc.metadata['source']}\n\nContent: {doc.metadata['content']}" for doc in retriever)
    return serialize_return

def main(user_id, thread_id, query):
    if not user_id:
        user_id = str(uuid.uuid4())
    if not thread_id:
        thread_id = str(uuid.uuid4())
        query = input("Please enter a query: ")
        if query.lower() in ["exit", "quit"]:
            return
        if not query:
            print("Please enter a query")
    agent = create_agent(
        model = llm,
        tools = [tavily_search, rag_retriever],
        state_schema= CustomAgent,
        checkpointer= checkpointer
            )

    full_prompt = prompt_template.format(query=query)

    init_state={"messages":[HumanMessage(content=full_prompt)]}
    config = {"configurable":{"thread_id":thread_id, "user_id":user_id}}

    agent_response = agent.invoke(init_state, config=config)

    return agent_response




if __name__ == "__main__":
    print("="*40)
    print("Welcome to the RAG Retriever!")
    u_id = input("Please enter a user ID: ")
    t_id = input("Please enter a thread ID: ")
    q = input("Please enter a query: ")
    print("AI is loading")
    response = main(u_id, t_id, q)
    print(f"Query: {q}\n\nAnswer: {response}")




