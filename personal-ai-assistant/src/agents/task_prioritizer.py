from src.core.llm_client import GeminiClient
import re

gemini = GeminiClient()

class TaskPrioritizer:
    def __init__(self):
        self.model = gemini

    def prioritize(self, tasks):
        """
        Takes a list of tasks and returns them ranked by urgency/importance.
        Each task can be a string (from emails, calendar events, or manual input).
        """
        if not tasks:
            return ["No tasks provided."]

        prompt = f"""
        You are a Task Prioritizer Agent.
        Given the following tasks, rank them by urgency and importance.
        Output as a numbered list, from most urgent to least urgent:

        Tasks:
        {chr(10).join(tasks)}
        """
        response = self.model.chat(prompt)
        # Clean up: split lines, drop blanks, keep numbered lines
        # Extract lines starting with a number (regex keeps "1. Task")
        lines = [re.sub(r"\*\*|\(.*?\)", "", line).strip() 
                 for line in response.splitlines() 
                 if re.match(r"^\d+\.", line.strip())]

        return lines
