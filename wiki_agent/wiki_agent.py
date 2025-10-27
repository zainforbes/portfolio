import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.wikipedia import WikipediaTools

# Load environment variables from .env file
load_dotenv()

def create_agent():
    """Creates and returns the Wikipedia research agent."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing GEMINI_API_KEY in .env file")

    # Initialize the agent (compatible with all stable Agno builds)
    agent = Agent(
        model=Gemini(id="gemini-2.0-flash", api_key=api_key),
        description="You are a researcher that finds and summarizes topics from Wikipedia.",
        tools=[WikipediaTools()],
        markdown=True
    )
    return agent


def run_agent():
    """Continuously prompts the user for Wikipedia topics and summarizes them."""
    agent = create_agent()
    print("\n🔍 Wikipedia Research Agent")
    print("Type 'exit/quit' to quit.\n")

    while True:
        topic = input("Enter a topic to search: ").strip()
        if topic.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        if not topic:
            print("⚠️ Please enter a valid topic.\n")
            continue

        print(f"\n📚 Searching Wikipedia for '{topic}'...\n")
        try:
            agent.print_response(
                f"Search Wikipedia for '{topic}' and summarize the main points.",
                markdown=True
            )
            print("\n" + "=" * 80 + "\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    run_agent()
