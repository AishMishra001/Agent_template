from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, OpenAIServerModel, load_tool, tool
import datetime
import requests
import requests
import pytz

import yaml
from tools.final_answer import FinalAnswerTool

from Gradio_UI import GradioUI

# Below is an example of a tool that does nothing. Amaze us with your creativity !
@tool
def my_custom_tool(arg1:str, arg2:int)-> str: #it's import to specify the return type
    #Keep this format for the description / args / args description but feel free to modify the tool
    """A tool that does nothing yet 
    Args:
        arg1: the first argument
        arg2: the second argument
    """
    return "What magic will you build ?"

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


@tool
def get_weather(location: str) -> str:
    """A tool that fetches the current live weather for a given city or location.
    
    Args:
        location: The name of the city or location (e.g., 'Paris', 'Tokyo', 'San Francisco').
    """
    try:
        # wttr.in with ?format=... gives a clean, concise one-line summary
        # %C = condition, %t = temp, %h = humidity, %w = wind
        url = f"https://wttr.in/{location}?format=%l:+%C+%t+(Humidity:+%h,+Wind:+%w)"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            return response.text.strip()
        else:
            return f"Failed to retrieve weather for '{location}'. Status code: {response.status_code}"
    except Exception as e:
        return f"Error retrieving weather for '{location}': {str(e)}"

final_answer = FinalAnswerTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = OpenAIServerModel(
    model_id="gemma4:e4b",
    api_base="http://localhost:11434/v1",
    api_key="ollama",
)


# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[final_answer,get_current_time_in_timezone,get_weather], ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()