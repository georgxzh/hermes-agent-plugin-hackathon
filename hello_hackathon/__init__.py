"""hello_hackathon — minimal Hermes plugin with one tool."""

import json

# What the LLM sees: name, when to use it, and its (empty) parameters.
HELLO_SCHEMA = {
    "name": "hello_hackathon",
    "description": (
        "Return a greeting from the hello_hackathon plugin. "
        "Call this when the user asks to test or say hello to the hackathon plugin."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}


def hello_hackathon(args: dict, **kwargs) -> str:
    """Tool handler: takes the LLM's args dict, always returns a JSON string."""
    try:
        return json.dumps({"message": "Hello from my plugin."})
    except Exception as e:  # handlers must never raise
        return json.dumps({"error": str(e)})


def register(ctx):
    """Called once by Hermes at startup to wire the schema to the handler."""
    ctx.register_tool(
        name="hello_hackathon",
        toolset="hello_hackathon",
        schema=HELLO_SCHEMA,
        handler=hello_hackathon,
    )
