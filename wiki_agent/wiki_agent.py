import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.wikipedia import WikipediaTools

# Load environment variables
load_dotenv()

def create_agent():
    """Creates and returns the Wikipedia research agent."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing GEMINI_API_KEY in .env file")

    agent = Agent(
        model=Gemini(id="gemini-2.0-flash", api_key=api_key),
        description="You are a researcher specialized in searching Wikipedia.",
        tools=[WikipediaTools()],
        markdown=True
    )
    return agent

def run_agent():
    """Runs the agent on a Wikipedia query."""
    agent = create_agent()
    response = agent.print_response(
        "Search Wikipedia for 'Time series analysis' and summarize the 3 main points",
        markdown=True
    )
    return response

if __name__ == "__main__":
    run_agent()
