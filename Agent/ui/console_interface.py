from agent.core.rag_application import RAGApplication

class ConsoleInterface:
    """Console-based user interface"""
    
    def __init__(self, app: RAGApplication):
        self.app = app
        self.exit_commands = {'exit', 'quit', 'q'}
    
    def display_menu(self):
        """Display the main menu"""
        print("\n" + "="*50)
        print("🤖 RAG Data Lineage System - Main Menu")
        print("="*50)
        print("1. 📄 Query documents")
        print("2. 📊 Show system statistics")
        print("3. 🔍 Search similar documents")
        print("4. 📈 Generate lineage report")
        print("5. 🎨 Visualize embeddings")  # New option
        print("6. ❓ Show help")
        print("7. 🚪 Exit")
        print("="*50)

    def run_interactive_session(self):
        """Run the interactive console session"""
        print("\n" + "="*50)
        print("🎯 Welcome to RAG Data Lineage System")
        print("="*50)

        while True:
            print("\nChoose an option:")
            print("1. 📄 Query documents")
            print("2. 🧠 Query with thought process")  # New option
            print("3. 📊 Show statistics")
            print("4. 🔍 Search similar documents")
            print("5. 📈 Generate lineage report")
            print("6. 🎨 Visualize embeddings")
            print("7. ❓ Show help")
            print("8. 🚪 Exit")

            choice = input("\nEnter your choice (1-8): ").strip()

            try:
                if choice == "1":
                    self.handle_document_query()
                elif choice == "2":
                    self.handle_verbose_query()  # New method
                elif choice == "3":
                    self.show_statistics()
                elif choice == "4":
                    self.search_similar_documents()
                elif choice == "5":
                    self.generate_lineage_report()
                elif choice == "6":
                    self.visualize_embeddings()
                elif choice == "7":
                    self.show_help()
                elif choice == "8" or choice.lower() in self.exit_commands:
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please try again.")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def handle_document_query(self):
        """Handle document query from user input"""
        try:
            if not self.app.is_ready():
                print("❌ System not ready. Please check the health status.")
                return

            print("\n" + "="*50)
            print("📄 Document Query")
            print("="*50)
            print("Enter your question (or 'back' to return to menu)")
            print("💡 Tip: Add '--verbose' or '-v' to see the agent's thought process")

            while True:
                question_input = input("\n🤔 Your question: ").strip()

                if question_input.lower() in {'back', 'menu'}:
                    break

                if question_input.lower() in self.exit_commands:
                    return

                if not question_input:
                    print("❌ Please enter a valid question.")
                    continue

                # Check for verbose flag
                verbose = False
                if question_input.endswith('--verbose') or question_input.endswith('-v'):
                    verbose = True
                    # Remove the verbose flag from the question
                    question = question_input.replace('--verbose', '').replace('-v', '').strip()
                else:
                    question = question_input

                try:
                    if not verbose:
                        print("\n🔄 Processing your question...")

                    answer = self.app.process_question(question, verbose=verbose)

                    if not verbose:
                        print("\n" + "="*50)
                        print("📋 Answer:")
                        print("="*50)
                    
                    print(answer)
                    
                    if not verbose:
                        print("="*50)

                    # Ask if user wants to ask another question
                    continue_choice = input("\nWould you like to ask another question? (y/n): ").strip().lower()
                    if continue_choice not in ['y', 'yes']:
                        break

                except Exception as e:
                    print(f"❌ Error processing question: {e}")

        except Exception as e:
            print(f"❌ Error in document query: {e}")
    
    def handle_verbose_query(self):
        """Handle queries with verbose thought process display"""
        try:
            if not self.app.is_ready():
                print("❌ System not ready. Please check the health status.")
                return

            print("\n" + "="*50)
            print("🧠 Query with Thought Process")
            print("="*50)
            print("Ask a question and see how the agent thinks through the problem!")

            while True:
                question = input("\n🤔 Your question (or 'back' to return): ").strip()

                if question.lower() in {'back', 'menu'}:
                    break

                if question.lower() in self.exit_commands:
                    return

                if not question:
                    print("❌ Please enter a valid question.")
                    continue

                try:
                    # Always use verbose mode for this option
                    answer = self.app.process_question(question, verbose=True)
                    
                    # Show the final answer clearly after the thought process
                    print("\n" + "="*60)
                    print("📋 FINAL ANSWER:")
                    print("="*60)
                    print(answer)
                    print("="*60)

                    # Ask if user wants to ask another question
                    continue_choice = input("\nWould you like to ask another question? (y/n): ").strip().lower()
                    if continue_choice not in ['y', 'yes']:
                        break

                except Exception as e:
                    print(f"❌ Error processing question: {e}")

        except Exception as e:
            print(f"❌ Error in verbose query: {e}")
    
    def visualize_embeddings(self):
        """Generate and display embedding visualizations"""
        try:
            from agent.visualization.embedding_visualizer import EmbeddingVisualizer
            
            print("\n🎨 Generating Embedding Visualizations...")
            
            # Create visualizer with cached model path
            visualizer = EmbeddingVisualizer(
                self.app.vector_store.chroma_client, 
                "data_lineage_queries_txt",
                self.app.config.embedding_model_path
            )
            
            # Load data
            data = visualizer.load_collection_data()
            if not data:
                print("❌ No embeddings found in collection")
                return
            
            print(f"📊 Found {data['count']} embeddings to visualize")
            
            # Ask user for visualization type
            print("\nVisualization options:")
            print("1. 2D t-SNE scatter plot (documents only)")
            print("2. 2D PCA scatter plot (documents only)") 
            print("3. 3D t-SNE scatter plot (documents only)")
            print("4. Similarity heatmap")
            print("5. 2D combined view (documents + questions)")
            print("6. 3D combined view (documents + questions)")
            print("7. All visualizations")
            
            choice = input("Select visualization type (1-7): ").strip()
            
            if choice == "1":
                fig = visualizer.create_2d_scatter_plot(method='tsne')
                fig.show()
            elif choice == "2":
                fig = visualizer.create_2d_scatter_plot(method='pca')
                fig.show()
            elif choice == "3":
                fig = visualizer.create_3d_scatter_plot(method='tsne')
                fig.show()
            elif choice == "4":
                fig = visualizer.create_cluster_heatmap()
                fig.show()
            elif choice == "5":
                # Combined 2D visualization
                questions = self.app.get_question_history()
                if not questions:
                    print("❌ No questions asked yet. Ask some questions first!")
                    return
                
                print(f"📝 Including {len(questions)} questions in visualization...")
                fig = visualizer.create_combined_visualization(
                    questions, method='tsne', n_components=2
                )
                if fig:
                    fig.show()
                    visualizer.save_visualization(fig, 'combined_2d_tsne', 'html')
                    
            elif choice == "6":
                # Combined 3D visualization
                questions = self.app.get_question_history()
                if not questions:
                    print("❌ No questions asked yet. Ask some questions first!")
                    return
                
                print(f"📝 Including {len(questions)} questions in visualization...")
                fig = visualizer.create_combined_visualization(
                    questions, method='tsne', n_components=3
                )
                if fig:
                    fig.show()
                    visualizer.save_visualization(fig, 'combined_3d_tsne', 'html')
                    
            elif choice == "7":
                # Generate all visualizations
                for method, dims in [('tsne', 2), ('pca', 2), ('tsne', 3)]:
                    if dims == 2:
                        fig = visualizer.create_2d_scatter_plot(method=method)
                    else:
                        fig = visualizer.create_3d_scatter_plot(method=method)
                    visualizer.save_visualization(fig, f'embeddings_{dims}d_{method}', 'html')
                
                fig_heatmap = visualizer.create_cluster_heatmap()
                visualizer.save_visualization(fig_heatmap, 'similarity_heatmap', 'png')
                
                # Combined visualizations if questions exist
                questions = self.app.get_question_history()
                if questions:
                    print(f"📝 Including {len(questions)} questions in combined visualizations...")
                    for dims in [2, 3]:
                        fig = visualizer.create_combined_visualization(
                            questions, method='tsne', n_components=dims
                        )
                        if fig:
                            visualizer.save_visualization(fig, f'combined_{dims}d_tsne', 'html')
                
                print("✅ All visualizations saved to 'visualizations' folder")
            
        except ImportError as e:
            print(f"❌ Missing required packages for visualization: {e}")
            print("Install with: pip install matplotlib seaborn plotly scikit-learn sentence-transformers")
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
    
    def run_single_query(self, question: str) -> str:
        """Run a single query"""
        if not self.app.is_ready():
            raise RuntimeError("Application is not ready")
        
        return self.app.process_question(question)
    
    def display_health_status(self):
        """Display application health status"""
        status = self.app.get_health_status()
        print("\n" + "="*40)
        print("🏥 Application Health Status")
        print("="*40)
        for key, value in status.items():
            status_icon = "✅" if value else "❌"
            print(f"{status_icon} {key}: {value}")
        print("="*40)

    def handle_query(self):
        """Handle user queries - main query interface"""
        self.handle_document_query()
    
    def show_statistics(self):
        """Show system statistics"""
        try:
            print("\n" + "="*50)
            print("📊 System Statistics")
            print("="*50)
            
            health = self.app.get_health_status()
            
            print(f"🟢 System Status: {'Ready' if self.app.is_ready() else 'Not Ready'}")
            print(f"📊 Total Tables: {health.get('total_tables', 0)}")
            print(f"🗄️  Vector Store: {'Ready' if health.get('vector_store_ready') else 'Not Ready'}")
            print(f"🤖 LLM Service: {'Available' if health.get('llm_available') else 'Not Available'}")
            print(f"📋 Schema Loaded: {'Yes' if health.get('schema_loaded') else 'No'}")
            
            # Show question history if available
            questions = self.app.get_question_history()
            print(f"❓ Questions Asked: {len(questions)}")
            
            if questions:
                print("\nRecent Questions:")
                for i, q in enumerate(questions[-5:], 1):  # Show last 5 questions
                    print(f"  {i}. {q[:80]}{'...' if len(q) > 80 else ''}")
            
        except Exception as e:
            print(f"❌ Error getting statistics: {e}")
    
    def handle_similarity_search(self):
        """Handle similarity search"""
        try:
            print("\n" + "="*50)
            print("🔍 Document Similarity Search")
            print("="*50)
            
            query = input("Enter search term: ").strip()
            if not query:
                print("❌ Please enter a valid search term.")
                return
            
            print(f"\n🔄 Searching for documents similar to: '{query}'")
            
            # Use the vector store to find similar documents
            results = self.app.vector_store.query(query, n_results=5)
            
            if results:
                print(f"\n📋 Found {len(results)} similar documents:")
                print("="*50)
                
                for i, result in enumerate(results, 1):
                    doc = result.get('document', '')
                    metadata = result.get('metadata', {})
                    
                    print(f"\n{i}. Document:")
                    print(f"   Content: {doc[:200]}{'...' if len(doc) > 200 else ''}")
                    if metadata:
                        print(f"   Metadata: {metadata}")
                    print("-" * 30)
            else:
                print("❌ No similar documents found.")
                
        except Exception as e:
            print(f"❌ Error in similarity search: {e}")
    
    def generate_lineage_report(self):
        """Generate a lineage report"""
        try:
            print("\n" + "="*50)
            print("📈 Data Lineage Report")
            print("="*50)
            
            if not self.app.schema_extractor:
                print("❌ Schema information not available.")
                return
            
            # Get all tables
            all_tables = self.app.schema_extractor.extract_tables()
            hive_tables = [t for t in all_tables if not self.app.schema_extractor.is_spark_table(t)]
            spark_tables = [t for t in all_tables if self.app.schema_extractor.is_spark_table(t)]
            
            print(f"📊 Total Tables: {len(all_tables)}")
            print(f"   🐝 Hive Tables: {len(hive_tables)}")
            print(f"   ⚡ Spark Tables: {len(spark_tables)}")
            
            # Show table relationships
            relationships = self.app.schema_extractor.get_table_relationships()
            if relationships:
                print(f"\n🔗 Table Relationships: {len(relationships)}")
                for table, rels in list(relationships.items())[:10]:  # Show first 10
                    upstream = rels.get('upstream', [])
                    downstream = rels.get('downstream', [])
                    print(f"   {table}:")
                    if upstream:
                        print(f"     ⬆️  Upstream: {', '.join(upstream)}")
                    if downstream:
                        print(f"     ⬇️  Downstream: {', '.join(downstream)}")
                
                if len(relationships) > 10:
                    print(f"   ... and {len(relationships) - 10} more relationships")
            
        except Exception as e:
            print(f"❌ Error generating lineage report: {e}")
    
    def show_help(self):
        """Show help information"""
        print("\n" + "="*50)
        print("❓ Help - RAG Data Lineage System")
        print("="*50)
        print("This system helps you query and analyze data lineage information.")
        print("\nAvailable Options:")
        print("1. 📄 Query documents - Ask questions about your data lineage")
        print("2. 📊 Show statistics - View system status and statistics")
        print("3. 🔍 Search similar documents - Find documents similar to your search term")
        print("4. 📈 Generate lineage report - View comprehensive lineage information")
        print("5. 🎨 Visualize embeddings - Create visual representations of your data")
        print("6. ❓ Show help - Display this help information")
        print("7. 🚪 Exit - Close the application")
        print("\nTips:")
        print("- Ask specific questions about tables, columns, or relationships")
        print("- Use natural language - the system understands context")
        print("- Try questions like 'What tables are connected to X?' or 'Show me all Spark tables'")
        print("="*50)