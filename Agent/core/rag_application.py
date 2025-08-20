from typing import Optional, List, Dict, Any
from datetime import datetime
from agent.config.settings import AppConfig
from agent.services.data_loaders import HiveDataLoader, SparkDataLoader
from agent.services.schema_extractors import HiveSchemaExtractor, SparkSchemaExtractor, UnifiedSchemaExtractor
from agent.services.llm_service import LlamaLLMService
from agent.services.vector_store import ChromaVectorStore
from agent.services.conversation_manager import ConversationManager

class RAGApplication:
    """Main RAG application orchestrator following SOLID principles"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.initialized = False
        self.question_history = []  # Add question history tracking for visualization

        # Initialize services
        self.hive_loader = HiveDataLoader()
        self.spark_loader = SparkDataLoader()
        self.vector_store = ChromaVectorStore(config)
        self.llm_service = LlamaLLMService(config)
        self.conversation_manager = ConversationManager()

        # Will be set during initialization
        self.schema_extractor: Optional[UnifiedSchemaExtractor] = None

    def initialize(self):
        """Initialize all application components"""
        try:
            print("Loading data...")
            hive_data = self.hive_loader.load_data(self.config.hive_file_path)
            spark_data = self.spark_loader.load_data(self.config.spark_file_path)

            all_data = hive_data + spark_data
            print(f"Loaded {len(all_data)} entries (Hive: {len(hive_data)}, Spark: {len(spark_data)})")

            print("Setting up vector store...")
            self.vector_store.setup(all_data)

            print("Extracting schemas...")
            hive_extractor = HiveSchemaExtractor(hive_data)
            spark_extractor = SparkSchemaExtractor(spark_data)
            self.schema_extractor = UnifiedSchemaExtractor(hive_extractor, spark_extractor)

            self.initialized = True

            all_tables = self.schema_extractor.extract_tables()
            hive_count = len([t for t in all_tables if not self.schema_extractor.is_spark_table(t)])
            spark_count = len([t for t in all_tables if self.schema_extractor.is_spark_table(t)])

            print(f"System ready! Tables: {len(all_tables)} total (Hive: {hive_count}, Spark: {spark_count})")

        except Exception as e:
            print(f"Failed to initialize application: {e}")
            raise

    def process_question(self, question: str, verbose: bool = False) -> str:
        """Process a user question using RAG principles with improved prompt engineering and filtering"""
        if not self.initialized:
            raise RuntimeError("Application not initialized. Call initialize() first.")

        if verbose:
            print("\n" + "="*60)
            print("🧠 AGENT THOUGHT PROCESS")
            print("="*60)

        # Track the question for visualization purposes
        self.question_history.append({
            'question': question,
            'timestamp': datetime.now().isoformat()
        })

        if verbose:
            print(f"📝 Original Question: '{question}'")

        resolved_question = self.conversation_manager.resolve_references(
            question,
            self.schema_extractor.extract_tables() if self.schema_extractor else set()
        )

        if verbose:
            if resolved_question != question:
                print(f"🔄 Resolved Question: '{resolved_question}'")
                print("   └─ Applied conversation context and reference resolution")
            else:
                print("✅ No reference resolution needed")

        if verbose:
            print(f"\n🔍 Step 1: Retrieving relevant documents from vector store...")
            print(f"   └─ Searching for top 5 most similar documents")

        # Retrieve more documents initially to allow for filtering
        all_relevant_results: List[Dict[str, Any]] = self.vector_store.query(resolved_question, n_results=5)

        if verbose:
            print(f"   ✅ Found {len(all_relevant_results)} candidate documents")

        # --- Filter out "scuffed" documents ---
        if verbose:
            print(f"\n🧹 Step 2: Filtering documents for quality...")

        cleaned_relevant_docs: List[str] = []
        scuff_patterns = [
            "Please let me know if this answer is correct or not.",
            "Thank you for your time and consideration.",
            "I'm looking forward to your feedback.",
            "Best regards,",
            "Sincerely,",
            "Regards,",
            "Please let me know if you need any further assistance."
        ]
        
        max_doc_length_for_context = 1500
        max_docs_in_context = 3

        filtered_count = 0
        for i, item in enumerate(all_relevant_results):
            doc_text = item.get('document', '')
            metadata = item.get('metadata', {})

            scuff_count = sum(1 for pattern in scuff_patterns if pattern.lower() in doc_text.lower())
            
            if verbose:
                print(f"   📄 Document {i+1}:")
                print(f"      └─ Length: {len(doc_text)} chars")
                print(f"      └─ Scuff patterns found: {scuff_count}")
                
            if len(doc_text) < max_doc_length_for_context and scuff_count < 2:
                cleaned_relevant_docs.append(doc_text)
                if verbose:
                    print(f"      ✅ INCLUDED (high quality)")
                    doc_preview = doc_text[:100] + "..." if len(doc_text) > 100 else doc_text
                    print(f"      └─ Preview: {doc_preview}")
            else:
                filtered_count += 1
                if verbose:
                    print(f"      ❌ FILTERED OUT (length: {len(doc_text)}, scuff: {scuff_count})")

            if len(cleaned_relevant_docs) >= max_docs_in_context:
                break

        if verbose:
            print(f"   📊 Final selection: {len(cleaned_relevant_docs)} documents, {filtered_count} filtered out")

        context_string = "\n".join(cleaned_relevant_docs)

        if verbose:
            print(f"\n📋 Step 3: Extracting relevant schema information...")

        schema_info = self._get_schema_context_for_llm()
        
        if verbose:
            schema_preview = schema_info[:200] + "..." if len(schema_info) > 200 else schema_info
            print(f"   ✅ Schema context prepared ({len(schema_info)} chars)")
            print(f"   └─ Preview: {schema_preview}")

        if verbose:
            print(f"\n🤖 Step 4: Building prompt for LLM...")
            total_context_chars = len(schema_info) + len(context_string) + len(resolved_question)
            estimated_tokens = int(total_context_chars / 4)  # Rough estimate
            print(f"   └─ Total context: ~{total_context_chars} characters (~{estimated_tokens} tokens)")

        final_prompt = (
            f"You are an expert data lineage assistant. Your goal is to provide precise, factual answers to user queries about data lineage.\n"
            f"You have access to structured schema information and relevant document context snippets.\n\n"
            f"--- SCHEMA INFORMATION ---\n"
            f"{schema_info}\n\n"
            f"--- RELEVANT DOCUMENT CONTEXT ---\n"
            f"{context_string}\n\n"
            f"--- USER QUESTION ---\n"
            f"{resolved_question}\n\n"
            f"--- INSTRUCTIONS ---\n"
            f"1.  **Answer Source:** Base your answer strictly on the USER QUESTION and the provided CONTEXT (SCHEMA INFORMATION and RELEVANT DOCUMENT CONTEXT).\n"
            f"2.  **Output Purity:** The FINAL OUTPUT must contain ONLY the direct answer to the user's question. \n"
            f"    *   Do NOT include any conversational filler, greetings, closings, apologies, meta-commentary, or instructional phrases (e.g., 'Thank you', 'Best regards', 'I am happy to help', 'Please let me know', 'Provide the direct answer ONLY').\n"
            f"    *   If the question is too vague, cannot be answered directly from the context, or requires specific follow-up information, respond ONLY with 'I cannot provide a direct answer based on the available context.'\n"
            f"3.  **Formatting:** \n"
            f"    a.  **Comprehensive Listing:** If the user asks to 'List all tables' or 'List all relationships', you MUST provide EVERY item in a numbered list. Do NOT summarize or use phrases like '...and X more'. List each table and each relationship individually.\n"
            f"    b.  **General Formatting:** If listing items (like columns), use a clear, numbered or bulleted list. Otherwise, provide a direct statement.\n\n"
            f"**Provide the FINAL answer ONLY:**\n"
        )

        if verbose:
            print(f"\n💭 Step 5: Querying LLM...")
            print(f"   └─ Model: {self.config.model_name}")
            print(f"   └─ Max tokens: 500")

        answer = self.llm_service.query(final_prompt, max_tokens=500)

        if verbose:
            print(f"   ✅ LLM response received ({len(answer)} characters)")
            print(f"\n🎯 Step 6: Final answer ready!")
            
        self.conversation_manager.add_conversation(question, answer)
        return answer

    def _get_schema_context_for_llm(self) -> str:
        """
        Generates a clean string representation of schema information for the LLM.
        """
        if not self.schema_extractor:
            return "No schema information available."

        schema_parts = []

        all_tables = sorted(list(self.schema_extractor.extract_tables()))
        if all_tables:
            schema_parts.append("Available Tables:")
            # Attempt to use a cleaner name if available from schema extraction logic
            for table in all_tables:
                # Consider using a method like self._clean_display_name(table) if you have one
                # For now, assuming table names from schema are already reasonably clean.
                schema_parts.append(f"- {table}")
            schema_parts.append("")

        # Limit the schema details to avoid overwhelming the LLM's context window
        # Showing columns for a few tables is generally sufficient.
        MAX_TABLES_FOR_COLUMNS = 5
        tables_for_columns = all_tables[:MAX_TABLES_FOR_COLUMNS]

        if tables_for_columns:
            for table in tables_for_columns:
                columns = self.schema_extractor.get_columns_for_table(table)
                if columns:
                    schema_parts.append(f"Columns for {table}:")
                    for col in columns:
                        schema_parts.append(f"  - {col}")
                    schema_parts.append("")

        relationships = self.schema_extractor.get_table_relationships()
        if relationships:
            schema_parts.append("Table Relationships:")
            # Limit relationships shown if there are too many
            MAX_RELATIONSHIPS_TO_SHOW = 10
            shown_rels = 0
            for table, rels in relationships.items():
                if shown_rels >= MAX_RELATIONSHIPS_TO_SHOW:
                    break
                upstream_str = ", ".join(rels.get('upstream', [])) or "None"
                downstream_str = ", ".join(rels.get('downstream', [])) or "None"
                schema_parts.append(f"- {table}: Upstream=[{upstream_str}], Downstream=[{downstream_str}]")
                shown_rels += 1
            if len(relationships) > MAX_RELATIONSHIPS_TO_SHOW:
                schema_parts.append(f"... and {len(relationships) - MAX_RELATIONSHIPS_TO_SHOW} more relationships.")
            schema_parts.append("")

        return "\n".join(schema_parts)

    def get_question_history(self) -> List[str]:
        """Get list of questions asked in this session"""
        return [q['question'] for q in self.question_history]
    
    def clear_question_history(self):
        """Clear the question history"""
        self.question_history = []

    def is_ready(self) -> bool:
        return (self.initialized and
                self.vector_store.is_ready() and
                self.llm_service.is_available() and
                self.schema_extractor is not None)

    def get_health_status(self) -> dict:
        return {
            "initialized": self.initialized,
            "vector_store_ready": self.vector_store.is_ready(),
            "llm_available": self.llm_service.is_available(),
            "schema_loaded": self.schema_extractor is not None,
            "total_tables": len(self.schema_extractor.extract_tables()) if self.schema_extractor else 0
        }
