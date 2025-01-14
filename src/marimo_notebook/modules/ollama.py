from typing import Iterator, Optional, Dict, Type, List
from llm import Model, Prompt, Response, Conversation
import ollama

class OllamaModel(Model):
    model_id: str
    needs_key: str = None  # Ollama doesn't need an API key
    key_env_var: str = None
    can_stream: bool = True

    def __init__(self, model_id: str = "mistral", **kwargs):
        self.model_id = model_id
        super().__init__(**kwargs)

    def execute(
        self,
        prompt: Prompt,
        stream: bool,
        response: Response,
        conversation: Optional[Conversation],
    ) -> Iterator[str]:
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

        # Access options directly instead of using .get()
        options = {
            "temperature": getattr(prompt.options, "temperature", 0.7),
            "num_predict": getattr(prompt.options, "max_tokens", None),
            "top_p": getattr(prompt.options, "top_p", 1.0),
        }

        completion = ollama.chat(
            model=self.model_id,
            messages=messages,
            stream=stream,
            options=options,
        )

        if stream:
            for chunk in completion:
                if chunk.get("message", {}).get("content"):
                    yield chunk["message"]["content"]
        else:
            yield completion["message"]["content"]

class Options(Model.Options):
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    # Add other Ollama-specific parameters as needed

SUPPORTED_MODELS = [model['name'] for model in ollama.list().get('models', [])]

_models: Dict[str, Type[OllamaModel]] = {}

def get_model(model_name: str) -> "OllamaModel":
    """
    Get an Ollama model instance by name.
    
    Args:
        model_name: Name of the Ollama model to use (e.g., "mistral", "llama2")
    
    Returns:
        OllamaModel: An instance of OllamaModel configured for the specified model
    
    Raises:
        ValueError: If the model name is not recognized
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models are: {', '.join(SUPPORTED_MODELS)}"
        )
    return OllamaModel(model_id=model_name)



