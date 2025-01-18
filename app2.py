import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from operator import itemgetter
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
import json

# Initialize Streamlit page configuration
st.set_page_config(page_title="E-commerce Assistant", layout="wide")

# Load environment variables
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not google_api_key or not tavily_api_key:
    raise ValueError("API keys are not set in the .env file.")

# PDF handling setup
pdf_path = "FAQ-ecommerce.pdf"
K = 3

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

@tool
def lookup_policy(query: str) -> str:
    """Consult the company policies to check whether certain options are permitted."""
    raw_text = extract_text_from_pdf(pdf_path)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", api_key=google_api_key
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_text(raw_text)
    vectorstore = FAISS.from_texts(texts, embeddings)

    docs = vectorstore.similarity_search(query, k=K)
    return "\n\n".join([doc.page_content for doc in docs])

# Tavily search setup
tavily = TavilySearchResults(max_results=2, api_key=tavily_api_key)

@tool
def search_tool(query: str) -> str:
    """Search the web to get the answer for the input query."""
    response = tavily.invoke({"query": query})
    return response

# Database setup
llm = HuggingFaceEndpoint(
    huggingfacehub_api_token="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",
    repo_id="mistralai/Mistral-Nemo-Instruct-2407",
    temperature=0.8
)

system_role = """Given the following user question, corresponding SQL query, and SQL result, answer the user question.\n
    Question: {question}\n
    SQL Query: {query}\n
    SQL Result: {result}\n
    Answer:
    """

db = SQLDatabase.from_uri("sqlite:///sample_ecommerce.db")
execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)
answer_prompt = PromptTemplate.from_template(system_role)
answer = answer_prompt | llm | StrOutputParser()

chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer
)

@tool
def query_sqldb(query: str) -> str:
    """Query the E-commerce Database and access all the company's information."""
    response = chain.invoke({"question": query})
    return response

# LangGraph setup
llm_tools = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None
)

tools = [lookup_policy, query_sqldb, search_tool]
llm_with_tools = llm_tools.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

class BasicToolNode:
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

tool_node = BasicToolNode(tools=[lookup_policy, query_sqldb, search_tool])
graph_builder.add_node("tools", tool_node)

def route_tools(state: State) -> Literal["tools", "__end__"]:
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"

graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", "__end__": "__end__"},
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}

# Streamlit UI
st.title("E-commerce Support Assistant")
st.markdown("---")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)
    elif isinstance(message, ToolMessage):
        with st.chat_message("tool"):
            try:
                content = json.loads(message.content)
                st.json(content)
            except json.JSONDecodeError:
                st.write(message.content)

# Chat input
if prompt := st.chat_input("How can I help you today?"):
    # Add user message to chat history
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = graph.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config
            )
            
            # Add bot response to chat history
            for message in response["messages"]:
                st.session_state.messages.append(message)
                
                if isinstance(message, AIMessage):
                    st.write(message.content)
                elif isinstance(message, ToolMessage):
                    try:
                        content = json.loads(message.content)
                        st.json(content)
                    except json.JSONDecodeError:
                        st.write(message.content)

# Sidebar
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    This chatbot can help you with:
    - Looking up company policies
    - Checking order status
    - Answering general e-commerce questions
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()