import os
from typing import Optional
from llama_cpp import Llama
from agent.interfaces.data_loader import LLMInterface
from agent.config.settings import AppConfig

class LlamaLLMService(LLMInterface):
    """Concrete implementation for Llama LLM service"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.llm: Optional[Llama] = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the Llama model"""
        if not os.path.exists(self.config.llama_model_path):
            raise FileNotFoundError(f"Llama model not found at {self.config.llama_model_path}")
        
        print(f"Initializing Llama model: {self.config.llama_model_path}")
        self.llm = Llama(
            model_path=self.config.llama_model_path,
            n_ctx=4096,
            n_gpu_layers=0,
            verbose=False,
            n_threads=4,
        )
        print("Llama model initialized successfully")
    
    def query(self, prompt: str, max_tokens: int = 300) -> str:
        """Query the LLM with the given prompt"""
        if not self.llm:
            return "LLM not available"
        
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                stop=["</s>", "[INST]"]
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return "Error generating response"
    
    def is_available(self) -> bool:
        """Check if LLM is available"""
        return self.llm is not None