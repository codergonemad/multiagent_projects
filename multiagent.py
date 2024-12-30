import streamlit as st
from typing import Annotated, Literal, Union
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

# Initialize tools
duckduckgo_tool = DuckDuckGoSearchRun()
repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code and do math. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str

# Define team members and system prompt
members = ["researcher", "coder"]
options = tuple(members + ["FINISH"])

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal["researcher", "coder", "FINISH"]

def initialize_llm():
    if 'GROQ_API_KEY' in st.secrets:
        return ChatGroq(groq_api_key=st.secrets['GROQ_API_KEY'], model_name="Gemma2-9b-It")
    else:
        st.error("Please set the GROQ_API_KEY in your Streamlit secrets")
        st.stop()

def supervisor_node(state: MessagesState) -> Command[Union[Literal["researcher", "coder"], Literal["__end__"]]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = "__end__"
    return Command(goto=goto)

def research_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = research_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="researcher")
            ]
        },
        goto="supervisor",
    )

def code_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = code_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="coder")
            ]
        },
        goto="supervisor",
    )

def initialize_graph():
    builder = StateGraph(MessagesState)
    builder.add_edge(START, "supervisor")
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", research_node)
    builder.add_node("coder", code_node)
    return builder.compile()

# Streamlit UI
st.title("AI Team Assistant")
st.write("Ask a question and our AI team (researcher and coder) will help you find the answer!")

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Initialize LLM and agents
llm = initialize_llm()
research_agent = create_react_agent(llm, tools=[duckduckgo_tool], state_modifier="You are a researcher. DO NOT do any math.")
code_agent = create_react_agent(llm, tools=[python_repl_tool])
graph = initialize_graph()

# User input
user_input = st.chat_input("Enter your question here...")

if user_input:
    # Add user message to history
    st.session_state.conversation_history.append(("user", user_input))
    
    # Process the input through the graph
    for step in graph.stream(
        {"messages": [("user", user_input)]}, subgraphs=True
    ):
        # Check if step is a tuple and extract relevant information
        if isinstance(step, tuple):
            node_name, state_dict = step
            
            # Only process if we have messages in the state
            if isinstance(state_dict, dict) and 'messages' in state_dict:
                for msg in state_dict['messages']:
                    if isinstance(msg, tuple) and len(msg) == 2:
                        role, content = msg
                        if role != "user":  # Avoid duplicating user messages
                            st.session_state.conversation_history.append((role, content))
        
        # Update the display immediately
        st.rerun()

# Display conversation history
for role, content in st.session_state.conversation_history:
    if role == "user":
        st.chat_message("user").write(content)
    else:
        with st.chat_message("assistant"):
            st.write(f"**{role.capitalize()}**: {content}")