import streamlit as st
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from dotenv import load_dotenv
import logging
import uuid
import os
import mapper  # Import mapper.py

load_dotenv()

st.set_page_config(page_title="Globot", page_icon="images/Elixirr_logo.png", layout="wide")

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Import the Athena function and tool from x.py
from x import query_athena_tool

try:
    from weathertools import get_current_weather, get_weather_forecast, get_historical_weather
    from weathertools2 import recommend_best_time_to_visit
    from wikipediatools import get_city_highlights, get_sport_clubs_info, get_sportsman_info
    from wikipediatools2 import get_best_travel_package, get_tourism_info
    from prediction_model import predict_tourism_growth, country_with_biggest_tourist_increase
    from map_draw import save_last_bot_response
    from mapper import  get_locations  # Updated imports
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    st.error(f"Error importing modules: {e}")

# Load environment variables

# Define tools
tools = [
    get_current_weather,
    get_weather_forecast,
    get_city_highlights,
    get_historical_weather,
    get_sport_clubs_info,
    get_sportsman_info,
    recommend_best_time_to_visit,
    get_best_travel_package,
    query_athena_tool,
    predict_tourism_growth,
    country_with_biggest_tourist_increase,
    get_tourism_info,
    save_last_bot_response, 
    get_locations
]

# Ensure AWS_DEFAULT_REGION is set
if 'AWS_DEFAULT_REGION' not in os.environ:
    os.environ['AWS_DEFAULT_REGION'] = 'eu-central-1'

# Initialize session state for memory and session ID
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Use StreamlitChatMessageHistory for persistent chat history
if 'message_history' not in st.session_state:
    st.session_state.message_history = StreamlitChatMessageHistory(key="chat_messages")

# Bind tools to model
chat_model_id = "anthropic.claude-3-haiku-20240307-v1:0"
chat_model = None
try:
    chat_model = ChatBedrock(model_id=chat_model_id).bind_tools(tools)
except Exception as e:
    logger.error(f"Error initializing ChatBedrock: {e}")
    st.error(f"Error initializing ChatBedrock: {e}")

# Create agent with updated prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant specializing in Olympic travel information. Use the tools at your disposal to answer questions. Maintain context from previous messages in the conversation. Remember details about the user that they've shared. If you're unsure about something, you can ask for clarification.
    For any user queries not related to travel, first use the 'query_athena' tool. When using this tool return the result of the query as a response to the user.
    Always respond to the user conversationally. Never mention tools, table names and queries, just answer the user's question.
    For example, if the user asked you to 'list all coaches from <some_country> at the 2024 Olympic Games', you would use the 'query_athena' tool and respond:
    'These are the coaches from <some_country>:
    <coach_1>,
    <coach_2>,
    <coach_3>
    etc.'"""
    ),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Create agent and executor
agent_executor = None
if chat_model:
    try:
        agent = create_tool_calling_agent(chat_model, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        with_message_history = RunnableWithMessageHistory(
            agent_executor,
            lambda session_id: st.session_state.message_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )
    except Exception as e:
        logger.error(f"Error creating agent and executor: {e}")
        st.error(f"Error creating agent and executor: {e}")

MAX_HISTORY_LENGTH = 5000  # Maximum number of characters to keep in history
MAX_INPUT_LENGTH = 1000   # Maximum length for user input

def truncate_history(messages, max_length):
    """Truncate the message history to a maximum length, keeping the most recent messages.""" 
    current_length = sum(len(get_message_content(msg)) for msg in messages)
    while current_length > max_length and messages:
        removed_message = messages.pop(0)
        current_length -= len(get_message_content(removed_message))
    return messages


def get_message_content(message):
    """Safely extract content from various message formats."""
    if isinstance(message, (HumanMessage, AIMessage)):
        return message.content
    elif isinstance(message, dict):
        return message.get('content', message.get('text', ''))
    elif isinstance(message, str):
        return message
    elif isinstance(message, list) and len(message) > 0 and isinstance(message[0], dict):
        return message[0].get('text', '')
    else:
        return str(message)

def get_memory_contents():
    """Extract and format memory contents from the message history, prioritizing the latest messages."""
    memory_contents = []
    for msg in st.session_state.message_history.messages:
        content = get_message_content(msg)
        if isinstance(msg, HumanMessage):
            memory_contents.append(f"Human: {content}")
        else:
            memory_contents.append(f"AI: {content}")
    return "\n".join(memory_contents)

def manage_message_history():
    """Ensure that the message history does not exceed the maximum length."""
    st.session_state.message_history.messages = truncate_history(
        st.session_state.message_history.messages, 
        MAX_HISTORY_LENGTH
    )

def handle_user_input(user_input):
    if not agent_executor:
        return "Agent executor is not initialized."
    
    # Truncate user input if it exceeds the maximum length
    if len(user_input) > MAX_INPUT_LENGTH:
        user_input = user_input[:MAX_INPUT_LENGTH] + "..."
    
    try:
        # Add user input to history
        st.session_state.message_history.add_message(HumanMessage(content=user_input))
        manage_message_history()

        # Prepare the chat history for the model
        chat_history = []
        for msg in st.session_state.message_history.messages:
            content = get_message_content(msg)
            if isinstance(msg, HumanMessage):
                chat_history.append((content, None))
            else:
                chat_history.append((None, content))
        
        # Extract memory contents and include it in the input
        memory_contents = get_memory_contents()
        full_input = f"{user_input}\n\nPrevious Conversation:\n{memory_contents}"
        
        # Invoke the agent with the updated memory
        response = with_message_history.invoke(
            {"input": full_input, "chat_history": chat_history},
            config={"configurable": {"session_id": st.session_state.session_id}}
        )
        logger.debug(f"Response from agent: {response}")
        
        # Extract and format the bot's response
        bot_response = ''
        if isinstance(response, dict) and 'output' in response:
            bot_response = response['output']
        elif isinstance(response, str):
            bot_response = response
        elif isinstance(response, list) and len(response) > 0 and isinstance(response[0], dict):
            bot_response = response[0].get('text', '')
        
        # Add bot response to history
        st.session_state.message_history.add_message(AIMessage(content=bot_response))
        manage_message_history()
        
    except Exception as e:
        logger.error(f"Error handling user input: {e}")
        return f"An error occurred: {str(e)}"
    
    return bot_response


# Display banner image

# Display banner image
st.image("images/banner2.png", use_column_width=True)
 
# Sidebar with bot image and introduction text
with st.sidebar:
    st.image("images/Elixirr_logo.png", width=100)
    st.markdown("""**Hello! I'm Globot, your friendly travel assistant for Olympic Games information. Ask me about travel destinations, weather, and more.**
 
**Example Requests:**
- What is the best time to visit Paris?               
- Tell me about the Olympic games in Sydney.
- How many medalists does Croatia have?
- What are the top tourist attractions in Tokyo?
- Get the names of all the current Olympic champions in the weightlifting events.
- Give top 3 countries with biggest increase in tourism for the next year
- Predict tourism growth of croatia for 2025.
- Save it to PDF
    """)
# Images for user and bot
user_image = "images/user.png"
bot_image = "images/Elixirr_logo.png"

# Function to display messages
def display_message(image_url, sender, message, is_user=True):
    col1, col2 = st.columns([1, 9])
    with col1:
        st.image(image_url, width=50)
    with col2:
        content = get_message_content(message)
        if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
            content = content[0].get('text', '')
        st.write(f"**{sender}:** {content}", unsafe_allow_html=True)




def on_suggestion_click(query):
    st.session_state.user_input = query
    response = handle_user_input(query)
    if response:
        display_message(bot_image, "Bot", response, is_user=False)
 
# User input with suggestion buttons
st.sidebar.header("Suggestion Buttons")
suggestions = [
    "Which countries are the most successful in their olympic performance?",
    "What are the trends in medal counts for athletics across Italy?",
    "Tell me about the Olympic records in swimming.",
    "Give me a travel itinerary for 15 days across Asian countries focusing on countries that did well in the Olympics",
]
 
# Display suggestion buttons
for suggestion in suggestions:
    if st.sidebar.button(suggestion):
        on_suggestion_click(suggestion)

# Text input for custom queries
st.text_input("Enter your query:", key="user_input", on_change=lambda: on_suggestion_click(st.session_state.user_input))
 
# Display the chat interface
if st.button("Send"):
    user_input = st.session_state.user_input
    if user_input:
        response = handle_user_input(user_input)
        if response:
            display_message(bot_image, "Bot", response, is_user=False)
 
 

 

