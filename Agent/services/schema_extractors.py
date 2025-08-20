import json
import re
from typing import List, Any, Set, Dict
from agent.interfaces.data_loader import SchemaExtractorInterface

class HiveSchemaExtractor(SchemaExtractorInterface):
    """Extract schema information from Hive data"""
    
    def __init__(self, hive_data: List[Any]):
        self.tables: Set[str] = set()
        self.columns: Dict[str, Set[str]] = {}
        self.table_relationships: Dict[str, Dict[str, Set[str]]] = {}
        self.raw_queries: List[str] = []
        
        self._extract_from_data(hive_data)
    
    def _extract_from_data(self, hive_data: List[Any]):
        """Extract schema from Hive data entries"""
        for entry in hive_data:
            if entry.get('metadata', {}).get('source_type') == 'hive':
                self._process_hive_entry(entry)
    
    def _process_hive_entry(self, entry: dict):
        """Process a single Hive entry"""
        metadata = entry.get('metadata', {})
        
        # Store raw queries
        query_text = metadata.get('query_text', '')
        if query_text:
            self.raw_queries.append(query_text)
        
        # Extract from vertices
        if 'vertices_text' in metadata:
            try:
                vertices = json.loads(metadata['vertices_text'])
                self._extract_from_vertices(vertices)
            except json.JSONDecodeError:
                pass
        
        # Extract relationships from SQL
        if query_text:
            self._extract_relationships(query_text)
    
    def _extract_from_vertices(self, vertices: List[dict]):
        """Extract tables and columns from vertices"""
        for vertex in vertices:
            if vertex.get('vertexType') == 'TABLE':
                table_name = self._extract_table_name(vertex.get('vertexId', ''))
                if table_name:
                    self.tables.add(table_name)
            
            elif vertex.get('vertexType') == 'COLUMN':
                vertex_id = vertex.get('vertexId', '')
                table_name, column_name = self._parse_column_id(vertex_id)
                if table_name and column_name:
                    self.tables.add(table_name)
                    if table_name not in self.columns:
                        self.columns[table_name] = set()
                    self.columns[table_name].add(column_name)
    
    def _extract_table_name(self, vertex_id: str) -> str:
        """Extract table name from vertex ID"""
        if not vertex_id:
            return ""
        
        if '@' in vertex_id:
            vertex_id = vertex_id.split('@')[0]
        
        if '.' in vertex_id:
            return vertex_id.split('.')[-1]
        
        return vertex_id
    
    def _parse_column_id(self, vertex_id: str) -> tuple:
        """Parse column vertex ID to extract table and column names"""
        if not vertex_id:
            return None, None
        
        if '@' in vertex_id:
            vertex_id = vertex_id.split('@')[0]
        
        parts = vertex_id.split('.')
        if len(parts) >= 3:
            return parts[-2], parts[-1]  # table, column
        elif len(parts) == 2:
            return parts[0], parts[1]
        
        return None, None
    
    def _extract_relationships(self, query_text: str):
        """Extract table relationships from SQL query"""
        if not query_text:
            return
        
        query_lower = query_text.lower()
        
        # Extract target table
        target_table = None
        insert_match = re.search(r'insert\s+(?:overwrite\s+)?table\s+(\w+)', query_lower)
        if insert_match:
            target_table = insert_match.group(1)
        
        create_match = re.search(r'create\s+table\s+(\w+)', query_lower)
        if create_match:
            target_table = create_match.group(1)
        
        if not target_table:
            return
        
        # Extract source tables
        from_matches = re.findall(r'from\s+([^\s\n,;]+)', query_lower)
        join_matches = re.findall(r'join\s+([^\s\n,;]+)', query_lower)
        
        source_tables = []
        for match in from_matches + join_matches:
            table_name = match.split('.')[-1] if '.' in match else match
            if table_name != target_table:
                source_tables.append(table_name)
        
        # Build relationships
        if source_tables:
            if target_table not in self.table_relationships:
                self.table_relationships[target_table] = {'upstream': set(), 'downstream': set()}
            
            for source_table in source_tables:
                self.table_relationships[target_table]['upstream'].add(source_table)
                
                if source_table not in self.table_relationships:
                    self.table_relationships[source_table] = {'upstream': set(), 'downstream': set()}
                self.table_relationships[source_table]['downstream'].add(target_table)
    
    def extract_tables(self) -> set:
        """Extract all table names"""
        return self.tables
    
    def get_columns_for_table(self, table_name: str) -> List[str]:
        """Get columns for a specific table"""
        return sorted(self.columns.get(table_name, set()))
    
    def get_table_relationships(self) -> dict:
        """Get table relationship mappings"""
        return self.table_relationships

class SparkSchemaExtractor(SchemaExtractorInterface):
    """Extract schema information from Spark data"""
    
    def __init__(self, spark_data: List[Any]):
        self.spark_table_paths: Set[str] = set()
        self.spark_columns: Dict[str, List[dict]] = {}
        self.spark_lineage: Dict[str, dict] = {}
        self.table_relationships: Dict[str, Dict[str, Set[str]]] = {}
        
        self._extract_from_data(spark_data)
    
    def _extract_from_data(self, spark_data: List[Any]):
        """Extract schema from Spark data entries"""
        for entry in spark_data:
            if entry.get('metadata', {}).get('source_type') == 'spark':
                self._process_spark_entry(entry)
    
    def _process_spark_entry(self, entry: dict):
        """Process a single Spark entry"""
        metadata = entry.get('metadata', {})
        
        # Extract inputs and outputs
        inputs = metadata.get('inputs', [])
        outputs = metadata.get('outputs', [])
        
        # Process inputs
        for inp in inputs:
            full_name = inp.get('full_name')
            if full_name:
                self.spark_table_paths.add(full_name)
                if full_name not in self.spark_columns:
                    self.spark_columns[full_name] = []
                
                columns = inp.get('columns', [])
                self.spark_columns[full_name].extend(columns)
        
        # Process outputs
        for out in outputs:
            full_name = out.get('full_name')
            if full_name:
                self.spark_table_paths.add(full_name)
                if full_name not in self.spark_columns:
                    self.spark_columns[full_name] = []
                
                columns = out.get('columns', [])
                self.spark_columns[full_name].extend(columns)
                
                # Store column lineage
                column_lineage = out.get('column_lineage', {})
                if column_lineage:
                    self.spark_lineage[full_name] = column_lineage
        
        # Extract relationships
        self._extract_spark_relationships(inputs, outputs)
    
    def _extract_spark_relationships(self, inputs: List[dict], outputs: List[dict]):
        """Extract relationships between inputs and outputs"""
        for out in outputs:
            output_name = out.get('full_name')
            if output_name:
                if output_name not in self.table_relationships:
                    self.table_relationships[output_name] = {'upstream': set(), 'downstream': set()}
                
                for inp in inputs:
                    input_name = inp.get('full_name')
                    if input_name:
                        self.table_relationships[output_name]['upstream'].add(input_name)
                        
                        if input_name not in self.table_relationships:
                            self.table_relationships[input_name] = {'upstream': set(), 'downstream': set()}
                        self.table_relationships[input_name]['downstream'].add(output_name)
    
    def extract_tables(self) -> set:
        """Extract all table names"""
        return self.spark_table_paths
    
    def get_columns_for_table(self, table_name: str) -> List[str]:
        """Get columns for a specific table"""
        columns = self.spark_columns.get(table_name, [])
        return [f"{col['name']} ({col['type']})" for col in columns if isinstance(col, dict)]
    
    def get_table_relationships(self) -> dict:
        """Get table relationship mappings"""
        return self.table_relationships

class UnifiedSchemaExtractor(SchemaExtractorInterface):
    """Unified schema extractor combining Hive and Spark"""
    
    def __init__(self, hive_extractor: HiveSchemaExtractor, spark_extractor: SparkSchemaExtractor):
        self.hive_extractor = hive_extractor
        self.spark_extractor = spark_extractor
    
    def extract_tables(self) -> set:
        """Extract all tables from both Hive and Spark"""
        return self.hive_extractor.extract_tables().union(self.spark_extractor.extract_tables())
    
    def get_columns_for_table(self, table_name: str) -> List[str]:
        """Get columns for a table from either Hive or Spark"""
        # Try Hive first
        hive_columns = self.hive_extractor.get_columns_for_table(table_name)
        if hive_columns:
            return hive_columns
        
        # Try Spark
        return self.spark_extractor.get_columns_for_table(table_name)
    
    def get_table_relationships(self) -> dict:
        """Get unified table relationships"""
        relationships = {}
        relationships.update(self.hive_extractor.get_table_relationships())
        
        # Merge Spark relationships
        spark_rels = self.spark_extractor.get_table_relationships()
        for table, rels in spark_rels.items():
            if table in relationships:
                relationships[table]['upstream'].update(rels['upstream'])
                relationships[table]['downstream'].update(rels['downstream'])
            else:
                relationships[table] = rels
        
        return relationships
    
    def is_spark_table(self, table_name: str) -> bool:
        """Check if a table is a Spark table"""
        return table_name in self.spark_extractor.extract_tables()
    
    def get_spark_lineage(self, table_name: str) -> dict:
        """Get Spark column lineage for a table"""
        return self.spark_extractor.spark_lineage.get(table_name, {})