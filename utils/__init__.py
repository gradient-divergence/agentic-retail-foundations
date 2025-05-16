from .env import load_project_dotenv  # noqa: F401
from .openai_utils import safe_chat_completion  # noqa: F401

# Automatically load project-level .env once utils is imported.
load_project_dotenv()
