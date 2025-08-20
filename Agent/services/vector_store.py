import json
from typing import List, Any, Tuple, Dict
import chromadb
from sentence_transformers import SentenceTransformer
from agent.interfaces.data_loader import VectorStoreInterface
from agent.config.settings import AppConfig
import os

class ChromaVectorStore(VectorStoreInterface):
    """Concrete implementation for ChromaDB vector store"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.collection = None
        self.embedding_model = None
        self.chroma_client = None

    def setup(self, data_entries: List[Any]) -> Tuple[Any, Any]:
        """Setup ChromaDB with data entries"""
        print("Setting up Embedding Model...")

        model_to_load = None
        if self.config.embedding_model_path and os.path.isdir(self.config.embedding_model_path):
            model_to_load = self.config.embedding_model_path
            print(f"Loading embedding model from local path: {model_to_load}")
        elif self.config.embedding_model_name:
            model_to_load = self.config.embedding_model_name
            print(f"Loading embedding model by name: {model_to_load} (may require download)")
        else:
            raise ValueError("No embedding model name or path configured.")

        try:
            self.embedding_model = SentenceTransformer(model_to_load)
            print("Embedding Model Ready.")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise e

        print(f"Setting up ChromaDB at: {self.config.chroma_db_path}...")
        self.chroma_client = chromadb.PersistentClient(path=self.config.chroma_db_path)

        try:
            if self.config.collection_name in [c.name for c in self.chroma_client.list_collections()]:
                print(f"Deleting existing collection '{self.config.collection_name}'...")
                self.chroma_client.delete_collection(name=self.config.collection_name)
        except Exception as e:
            print(f"Warning during collection cleanup: {e}")

        self.collection = self.chroma_client.get_or_create_collection(name=self.config.collection_name)

        print(f"Adding {len(data_entries)} documents to ChromaDB...")
        documents, metadatas, ids = self._prepare_documents(data_entries)

        if documents:
            try:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                print("ChromaDB setup complete. Documents added.")
            except Exception as e:
                print(f"Error adding to ChromaDB: {e}")
                raise e
        else:
            print("No documents to add to ChromaDB.")

        return self.collection, self.embedding_model

    def _prepare_documents(self, data_entries: List[Any]) -> Tuple[List[str], List[dict], List[str]]:
        """Prepare documents for ChromaDB"""
        documents = []
        metadatas = []
        ids = []

        for i, entry in enumerate(data_entries):
            # Ensure IDs are unique. Using index 'i' if 'id' is missing or not unique.
            doc_id = entry.get("id", f"doc_{i}")
            doc_text = entry.get("text", "")
            original_metadata = entry.get("metadata", {}).copy()

            # Transform metadata for ChromaDB compatibility
            transformed_metadata = {}
            for key, value in original_metadata.items():
                if isinstance(value, (list, dict)):
                    try:
                        # Stringify complex types, truncate if too long
                        str_value = json.dumps(value)
                        if len(str_value) > 8000: # ChromaDB limits
                            str_value = str_value[:8000] + "...[truncated]"
                        transformed_metadata[key] = str_value
                    except TypeError: # Handle cases where json.dumps might fail
                        transformed_metadata[key] = str(value) # Fallback to string conversion
                elif value is None:
                    transformed_metadata[key] = ""
                else:
                    str_value = str(value)
                    if len(str_value) > 8000:  # ChromaDB limits
                        str_value = str_value[:8000] + "...[truncated]"
                    transformed_metadata[key] = str_value

            documents.append(doc_text)
            metadatas.append(transformed_metadata)
            ids.append(doc_id)

        return documents, metadatas, ids

    def query(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Query the vector store, returning a list of dictionaries,
        where each dict contains 'document' and 'metadata'.
        """
        if not self.collection:
            raise RuntimeError("Vector store not initialized. Call setup() first.")

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas'] # Request both documents and their metadata
            )

            formatted_results: List[Dict[str, Any]] = []
            # ChromaDB results structure is often nested: results['documents'][0], results['metadatas'][0]
            if results and results.get('documents') and results.get('metadatas'):
                # Ensure the lengths match before iterating
                num_results = min(len(results['documents'][0]), len(results['metadatas'][0]))
                for i in range(num_results):
                    formatted_results.append({
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i]
                    })
            return formatted_results

        except Exception as e:
            print(f"Error querying vector store: {e}")
            return []

    def is_ready(self) -> bool:
        """Check if the vector store is ready"""
        return self.collection is not None and self.embedding_model is not None