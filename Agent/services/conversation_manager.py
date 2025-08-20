import time
from typing import List, Dict, Any

class ConversationManager:
    """Manage conversation history and context"""
    
    def __init__(self, max_history: int = 5):
        self.conversation_history: List[Dict[str, Any]] = []
        self.max_history = max_history
    
    def add_conversation(self, question: str, answer: str):
        """Add a conversation to history"""
        self.conversation_history.append({
            'question': question,
            'answer': answer,
            'timestamp': time.time()
        })
        
        # Keep only last N conversations
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def resolve_references(self, question: str, schema_tables: set) -> str:
        """Resolve conversational references like 'the first one', 'that table'"""
        if not self.conversation_history:
            return question
        
        question_lower = question.lower()
        last_conversation = self.conversation_history[-1]
        last_answer = last_conversation['answer']
        
        # Handle "the first one", "first table"
        if any(phrase in question_lower for phrase in ['first one', 'first table', 'the first']):
            if "• " in last_answer:
                lines = last_answer.split('\n')
                for line in lines:
                    if line.strip().startswith('• '):
                        first_table = line.strip()[2:].strip()
                        question = question_lower.replace('the first one', first_table)
                        question = question.replace('first one', first_table)
                        question = question.replace('the first', first_table)
                        question = question.replace('first table', first_table)
                        return question
        
        # Handle "that table", "this table"
        if any(phrase in question_lower for phrase in ['that table', 'this table', 'it']):
            for table in schema_tables:
                if table.lower() in last_answer.lower():
                    question = question_lower.replace('that table', table)
                    question = question.replace('this table', table)
                    question = question.replace(' it ', f' {table} ')
                    return question
        
        return question
    
    def get_context(self, num_conversations: int = 2) -> str:
        """Get recent conversation context"""
        if not self.conversation_history:
            return ""
        
        recent = self.conversation_history[-num_conversations:]
        context_parts = []
        
        for conv in recent:
            q_snippet = conv['question'][:50] + "..." if len(conv['question']) > 50 else conv['question']
            a_snippet = conv['answer'][:100] + "..." if len(conv['answer']) > 100 else conv['answer']
            context_parts.append(f"Q: {q_snippet} A: {a_snippet}")
        
        return "\n".join(context_parts)