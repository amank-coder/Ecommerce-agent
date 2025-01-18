import os
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
from langchain_core.messages import HumanMessage, AIMessage
from operator import itemgetter

###########################################################################

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not google_api_key or not tavily_api_key:
    raise ValueError("API keys are not set in the .env file.")

######################################################################

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


# response = lookup_policy.invoke("How do I cancel my order?")
# print(response)

#######################################################################

tavily = TavilySearchResults(max_results=2, api_key=tavily_api_key)

@tool
def search_tool(query: str) -> str:
    """Search the web to get the answer for the input query. Input should be a search query."""
    print(f"Search query: {query}")  # Debug print
    response = tavily.invoke({"query": query})
    return response

# print(search_tool("Why use langgraph instead of langchain?"))

##########################################################################


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

# print(db.dialect)
# print(db.get_usable_table_names())
# db.run("SELECT * FROM Users LIMIT 10;")

execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(
    llm, db)
answer_prompt = PromptTemplate.from_template(
    system_role)


answer = answer_prompt | llm | StrOutputParser()
chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer
)

@tool
def query_sqldb(query: str)->str:

    """Query the E-commerce Database and access all the company's information. Input should be a search query."""
    # print(query)
    response = chain.invoke({"question": query})

    # print(response)
    return response

# message = "what is the status of order for user id 16?"
# response = query_sqldb.invoke(message)
# print(response)

#########################################################################

llm_tools = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None
)

tools = [lookup_policy, query_sqldb, search_tool]

llm_with_tools = llm_tools.bind_tools(tools)

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

import json
from langchain_core.messages import ToolMessage


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage.
    This class retrieves tool calls from the most recent AIMessage in the input
    and invokes the corresponding tool to generate responses."""

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

from typing import Literal


def route_tools(
    state: State,
) -> Literal["tools", "__end__"]:
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"


# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "__end__" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", "__end__": "__end__"},
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

config = {"configurable": {"thread_id": "1"}}

#############################################################

# user_input = "Hi there! My name is Farzad."

# # The config is the **second positional argument** to stream() or invoke()!
# events = graph.stream(
#     {"messages": [("user", user_input)]}, config, stream_mode="values"
# )
# for event in events:
#     event["messages"][-1].pretty_print()



# user_input = "Can I cancel my ticket 10 hours before the flight?"

# events = graph.stream(
#     {"messages": [("user", user_input)]}, config, stream_mode="values"
# )
# for event in events:
#     event["messages"][-1].pretty_print()


# user_input = "Right now Harris vs. Trump Presidential Debate is being boradcasted. I want the youtube link to this debate"

# events = graph.stream(
#     {"messages": [("user", user_input)]}, config, stream_mode="values"
# )
# for event in events:
#     event["messages"][-1].pretty_print()


###################################################################

from langchain_core.messages import AIMessage

user_input = input()

final_state = graph.invoke(
         {"messages": [HumanMessage(content=user_input)]},
         config=config
)

# Extract and print the last AIMessage content
print(final_state["messages"][-1].content)


