from abc import ABC, abstractmethod
from typing import List, Any, Optional

class DataLoaderInterface(ABC):
    """Interface for data loading operations"""
    
    @abstractmethod
    def load_data(self, file_path: str) -> List[Any]:
        """Load data from a file path"""
        pass

class SchemaExtractorInterface(ABC):
    """Interface for schema extraction operations"""
    
    @abstractmethod
    def extract_tables(self) -> set:
        """Extract all table names"""
        pass
    
    @abstractmethod
    def get_columns_for_table(self, table_name: str) -> List[str]:
        """Get columns for a specific table"""
        pass
    
    @abstractmethod
    def get_table_relationships(self) -> dict:
        """Get table relationship mappings"""
        pass

class LLMInterface(ABC):
    """Interface for LLM operations"""
    
    @abstractmethod
    def query(self, prompt: str, max_tokens: int = 300) -> str:
        """Query the LLM with a prompt"""
        pass

class VectorStoreInterface(ABC):
    """Interface for vector store operations"""
    
    @abstractmethod
    def setup(self, data_entries: List[Any]) -> Any:
        """Setup vector store with data entries"""
        pass
    
    @abstractmethod
    def query(self, query: str, n_results: int = 10) -> List[Any]:
        """Query the vector store"""
        pass