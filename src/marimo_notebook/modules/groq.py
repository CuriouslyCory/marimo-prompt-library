from typing import Iterator, Optional, Dict, Type, List
from llm import Model, Prompt, Response, Conversation
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

class GroqModel(Model):
    model_id: str
    needs_key: str = "Groq API key"
    key_env_var: str = "GROQ_API_KEY"
    can_stream: bool = True

    def __init__(self, model_id: str = "mixtral-8x7b-32768", **kwargs):
        self.model_id = model_id
        self.client = None
        super().__init__(**kwargs)

    def _ensure_client(self):
        if self.client is None:
            if not self.key:
                raise ValueError("Groq API key is required")
            self.client = Groq(api_key=self.key)

    def execute(
        self,
        prompt: Prompt,
        stream: bool,
        response: Response,
        conversation: Optional[Conversation],
    ) -> Iterator[str]:
        self._ensure_client()

        messages = []
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})
        
        # Handle conversation history if present
        if conversation and conversation.messages:
            for msg in conversation.messages:
                role = "assistant" if msg.type == "response" else "user"
                messages.append({"role": role, "content": msg.prompt.prompt})

        # Add the current prompt
        messages.append({"role": "user", "content": prompt.prompt})

        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            stream=stream,
            temperature=prompt.options.get("temperature", 0.7),
            max_tokens=prompt.options.get("max_tokens", None),
        )

        if stream:
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        else:
            yield completion.choices[0].message.content

        # Store usage information if available
        if hasattr(completion, "usage"):
            response.prompt_tokens = completion.usage.prompt_tokens
            response.completion_tokens = completion.usage.completion_tokens

class Options(Model.Options):
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    # Add other Groq-specific parameters as needed

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
SUPPORTED_MODELS = [
    model.id for model in client.models.list().data
]
client = None;

_models: Dict[str, Type[GroqModel]] = {}

def get_model(model_name: str) -> "GroqModel":
    """
    Get a Groq model instance by name.
    
    Args:
        model_name: Name of the Groq model to use (e.g., "mixtral-8x7b-32768", "llama2-70b-4096")
    
    Returns:
        GroqModel: An instance of GroqModel configured for the specified model
    
    Raises:
        ValueError: If the model name is not recognized
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models are: {', '.join(SUPPORTED_MODELS)}"
        )
    return GroqModel(model_id=model_name)



