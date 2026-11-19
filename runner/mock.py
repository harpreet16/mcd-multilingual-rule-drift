"""A fake model for tests and for trying the pipeline without a GPU."""

def make_mock_chat():
    """Return a chat(messages) function that gives a short made-up reply, never cut off."""
    def chat(messages):
        user_turns = sum(1 for m in messages if m["role"] == "user")
        return f"Let's talk about that. This is mock reply number {user_turns}.", False
    return chat
