"""
Main entry point for the RAG Data Lineage System
"""
from agent.config.settings import AppConfig
from agent.core.rag_application import RAGApplication
from agent.ui.console_interface import ConsoleInterface

def create_application(config: AppConfig = None) -> RAGApplication:
    """Factory function to create application with dependency injection"""
    if config is None:
        config = AppConfig.from_env()
    
    return RAGApplication(config)

def main():
    """Main entry point"""
    try:
        print("🚀 Starting RAG Data Lineage System...")
        
        # Create configuration
        config = AppConfig.from_env()
        
        # Create and initialize application
        app = create_application(config)
        app.initialize()
        
        # Create and run user interface
        ui = ConsoleInterface(app)
        ui.display_health_status()
        ui.run_interactive_session()
        
    except FileNotFoundError as e:
        print(f"❌ File error: {e}")
        print("Please ensure all required files are present.")
    except RuntimeError as e:
        print(f"❌ Runtime error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()