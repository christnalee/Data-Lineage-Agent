import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class AppConfig:
    """Application configuration following Single Responsibility Principle"""
    
    # File paths
    hive_file_path: str = "extracted_hive.txt"
    spark_file_path: str = "extracted_spark.txt"
    chroma_db_path: str = "./chroma_db_lineage_txt_2"
    collection_name: str = "data_lineage_queries_txt"
    llama_model_path: str = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
    embedding_model_path: str = "huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
    model_name: str = "llama3.2"
    
    # Environment settings
    transformers_offline: bool = True
    hf_datasets_offline: bool = True
    hf_hub_offline: bool = True
    
    def __post_init__(self):
        """Set environment variables after initialization"""
        if self.transformers_offline:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if self.hf_datasets_offline:
            os.environ["HF_DATASETS_OFFLINE"] = "1"
        if self.hf_hub_offline:
            os.environ["HF_HUB_OFFLINE"] = "1"

    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Create configuration from environment variables"""
        model_name = os.getenv('MODEL_NAME', 'llama3.2')
        
        return cls(
            hive_file_path=os.getenv('HIVE_FILE_PATH', cls.hive_file_path),
            spark_file_path=os.getenv('SPARK_FILE_PATH', cls.spark_file_path),
            chroma_db_path=os.getenv('CHROMA_DB_PATH', cls.chroma_db_path),
            collection_name=os.getenv('COLLECTION_NAME', cls.collection_name),
            llama_model_path=os.getenv('LLAMA_MODEL_PATH', cls.llama_model_path),
            embedding_model_path=os.getenv('EMBEDDING_MODEL_PATH', cls.embedding_model_path),
            model_name=model_name
        )