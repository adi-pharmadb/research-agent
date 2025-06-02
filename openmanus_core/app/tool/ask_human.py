from .base import BaseTool


_ASK_HUMAN_DESCRIPTION = """Asks the user for input and returns their response."""


class AskHuman(BaseTool):
    """Add a tool to ask human for help."""

    name: str = "ask_human"
    description: str = _ASK_HUMAN_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask the user.",
            },
        },
        "required": ["question"],
    }

    async def execute(self, question: str) -> str:
        """Ask the user a question and return their response."""
        response = input(f"\n{question}\nYour response: ")
        return response
