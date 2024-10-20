
import os
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core import load_index_from_storage
from llama_index.core import Settings
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import FunctionTool
from pydantic import Field
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = AzureOpenAI(
        deployment_name="dev-gpt-4o",
        api_key=os.environ.get("OPENAI_AZURE_API_KEY"),
        azure_endpoint='https://sonderwestus.openai.azure.com/',
        api_version="2024-02-01",
        model="gpt-4o",
    )

Settings.llm = llm

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-3-large",
    deployment_name="prod-text-embedding-3-large",
    api_key=os.environ.get("AZURE_OPENAI_API_KEY_EMBEDDING"),
    azure_endpoint='https://sondereastus.openai.azure.com/',
    api_version="2023-05-15",
)
Settings.embed_model = embed_model

persist_dir = "./rag_store"
storage_context = StorageContext.from_defaults(persist_dir=persist_dir)

# create vector store index
index = cur_index = load_index_from_storage(
    storage_context,
)

engine_tool = QueryEngineTool(
    query_engine=index.as_query_engine(),
    metadata=ToolMetadata(
        name=f"vector_index_faq",
        description=(
            "Useful for when you want to answer and questions that are likely in the FAQ Complete List and Medicare_FAQ_for_LPC_and_LMFT_Credentialing.pdf documents"
        ),
    )
)

def change_account(account_info: str = 
                   Field(..., description="Information to update the account with, such as name, address, etc.")) -> str:
    """Useful to change account information."""
    print("****** changing account ******")
    return f"We are changing your account information to {account_info}"

change_account_tool = FunctionTool.from_defaults(change_account)

tools = [engine_tool, change_account_tool]
system_prompt = """
You are a helpful assistant that can answer questions about the FAQ Complete List and Medicare_FAQ_for_LPC_and_LMFT_Credentialing.pdf documents.
You can also change account information if needed.
Stick to the the information provided in the documents and do not make up information.
If asked anything outside of the information provided in the documents, say you do not know and that people can use ChatGPT to answer those questions.
"""


########################### The streamlit code is right here below #################################

st.title("SonderMind - Agent with RAG")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if 'root_memory' not in st.session_state:
    st.session_state.root_memory = ChatMemoryBuffer.from_defaults(token_limit=8000)
else:
    root_memory = st.session_state.root_memory

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Create agent instance (assuming the necessary imports and definitions are present)
agent = OpenAIAgent.from_tools(system_prompt=system_prompt, streaming=True )

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate agent response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        current_history = root_memory.get()

        response = agent.stream_chat(user_input, chat_history=current_history)
        
        # Stream the response
        for token in response.response_gen:
            full_response += token
            response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)

        new_history = agent.memory.get_all()
        root_memory.set(new_history)
        st.session_state.root_memory = root_memory
    
    # Add agent response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})