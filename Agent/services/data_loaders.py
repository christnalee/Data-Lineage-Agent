import os
import json
import re
from typing import List, Any
from agent.interfaces.data_loader import DataLoaderInterface

class HiveDataLoader(DataLoaderInterface):
    """Concrete implementation for loading Hive data"""
    
    def load_data(self, file_path: str) -> List[Any]:
        """Load Hive data from text file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Hive file '{file_path}' not found")

        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        entries = content.split('-' * 50)
        for i, entry in enumerate(entries):
            if not entry.strip():
                continue

            doc_entry = {
                'id': f"doc_{i}",
                'text': entry.strip(),
                'metadata': {'source_type': 'hive'}
            }

            # Extract query text
            query_match = re.search(r'QUERY:\s*(.*?)(?=\n\nVERTICES:|$)', entry, re.DOTALL)
            if query_match:
                doc_entry['metadata']['query_text'] = query_match.group(1).strip()

            # Extract vertices
            vertices_match = re.search(r'VERTICES:\s*(.*?)(?=\n\n-|$)', entry, re.DOTALL)
            if vertices_match:
                doc_entry['metadata']['vertices_text'] = vertices_match.group(1).strip()

            if doc_entry['metadata']:
                data.append(doc_entry)
        
        return data

class SparkDataLoader(DataLoaderInterface):
    """Concrete implementation for loading Spark data"""
    
    def load_data(self, file_path: str) -> List[Any]:
        """Load Spark data from text file"""
        if not os.path.exists(file_path):
            print(f"Warning: Spark file '{file_path}' not found. Continuing without Spark data.")
            return []

        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        entries = content.split('-' * 50)
        for i, entry in enumerate(entries):
            if not entry.strip():
                continue

            doc_entry = {
                'id': f"spark_doc_{i}",
                'text': entry.strip(),
                'metadata': {'source_type': 'spark'}
            }

            # Parse Spark-specific metadata
            spark_metadata = self._parse_spark_entry(entry)
            doc_entry['metadata'].update(spark_metadata)

            if doc_entry['metadata']:
                data.append(doc_entry)
        
        return data
    
    def _parse_spark_entry(self, entry: str) -> dict:
        """Parse a single Spark entry to extract structured metadata"""
        metadata = {}
        
        # Extract basic job information
        patterns = {
            'event_type': r'EVENT TYPE:\s*(.+)',
            'event_time': r'EVENT TIME:\s*(.+)', 
            'job_name': r'JOB NAME:\s*(.+)',
            'run_id': r'RUN ID:\s*(.+)',
            'sql_query': r'SQL QUERY:\s*(.*?)(?=\nINPUTS:|$)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, entry, re.DOTALL if key == 'sql_query' else 0)
            if match:
                metadata[key] = match.group(1).strip()
        
        # Parse inputs and outputs
        inputs_section = re.search(r'INPUTS:\s*(.*?)(?=\nOUTPUTS:|$)', entry, re.DOTALL)
        if inputs_section:
            metadata['inputs'] = self._parse_spark_inputs(inputs_section.group(1))
        
        outputs_section = re.search(r'OUTPUTS:\s*(.*?)$', entry, re.DOTALL)
        if outputs_section:
            metadata['outputs'] = self._parse_spark_outputs(outputs_section.group(1))
        
        return metadata
    
    def _parse_spark_inputs(self, inputs_text: str) -> list:
        """Parse the INPUTS section of a Spark entry"""
        inputs = []
        
        if not inputs_text or inputs_text.strip() == "":
            return inputs
        
        # Split by "Input X:" pattern to get individual inputs
        input_blocks = re.split(r'\n\s*Input \d+:', inputs_text)
        
        for block in input_blocks:
            if not block.strip():
                continue
                
            input_data = {}
            
            # Extract Full Name
            full_name_match = re.search(r'Full Name:\s*(.+)', block)
            if full_name_match:
                input_data['full_name'] = full_name_match.group(1).strip()
            
            # Extract Namespace
            namespace_match = re.search(r'Namespace:\s*(.+)', block)
            if namespace_match:
                input_data['namespace'] = namespace_match.group(1).strip()
            
            # Extract Name
            name_match = re.search(r'Name:\s*(.+)', block)
            if name_match:
                input_data['name'] = name_match.group(1).strip()
            
            # Extract Columns
            columns_section = re.search(r'Columns:\s*(.*?)(?=\n\s*Symlinks:|$)', block, re.DOTALL)
            if columns_section:
                input_data['columns'] = self._parse_columns(columns_section.group(1))
            
            # Extract Symlinks
            symlinks_section = re.search(r'Symlinks:\s*(.*?)$', block, re.DOTALL)
            if symlinks_section:
                input_data['symlinks'] = self._parse_symlinks(symlinks_section.group(1))
            
            if input_data:  # Only add if we found some data
                inputs.append(input_data)
        
        return inputs
    
    def _parse_spark_outputs(self, outputs_text: str) -> list:
        """Parse the OUTPUTS section of a Spark entry"""
        outputs = []
        
        if not outputs_text or outputs_text.strip() == "":
            return outputs
        
        # Split by "Output X:" pattern to get individual outputs
        output_blocks = re.split(r'\n\s*Output \d+:', outputs_text)
        
        for block in output_blocks:
            if not block.strip():
                continue
                
            output_data = {}
            
            # Extract Full Name
            full_name_match = re.search(r'Full Name:\s*(.+)', block)
            if full_name_match:
                output_data['full_name'] = full_name_match.group(1).strip()
            
            # Extract Namespace
            namespace_match = re.search(r'Namespace:\s*(.+)', block)
            if namespace_match:
                output_data['namespace'] = namespace_match.group(1).strip()
            
            # Extract Name
            name_match = re.search(r'Name:\s*(.+)', block)
            if name_match:
                output_data['name'] = name_match.group(1).strip()
            
            # Extract Columns
            columns_section = re.search(r'Columns:\s*(.*?)(?=\n\s*Column Lineage:|$)', block, re.DOTALL)
            if columns_section:
                output_data['columns'] = self._parse_columns(columns_section.group(1))
            
            # Extract Column Lineage
            lineage_section = re.search(r'Column Lineage:\s*(.*?)$', block, re.DOTALL)
            if lineage_section:
                output_data['column_lineage'] = self._parse_column_lineage(lineage_section.group(1))
            
            if output_data:  # Only add if we found some data
                outputs.append(output_data)
        
        return outputs
    
    def _parse_columns(self, columns_text: str) -> list:
        """Parse the columns section"""
        columns = []
        
        if not columns_text:
            return columns
        
        # Split by lines and process each column
        lines = columns_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('- '):
                # Remove the "- " prefix
                column_def = line[2:].strip()
                
                # Parse column name and type: "column_name (type)"
                match = re.match(r'(\w+)\s*\(([^)]+)\)', column_def)
                if match:
                    columns.append({
                        'name': match.group(1),
                        'type': match.group(2)
                    })
                else:
                    # Fallback for columns without type info
                    columns.append({
                        'name': column_def,
                        'type': 'unknown'
                    })
        
        return columns
    
    def _parse_symlinks(self, symlinks_text: str) -> list:
        """Parse the symlinks section"""
        symlinks = []
        
        if not symlinks_text:
            return symlinks
        
        # Split by lines and process each symlink
        lines = symlinks_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('- '):
                # Remove the "- " prefix
                symlink_def = line[2:].strip()
                
                # Parse symlink: "url (TYPE)"
                match = re.match(r'(.+?)\s*\(([^)]+)\)', symlink_def)
                if match:
                    symlinks.append({
                        'url': match.group(1).strip(),
                        'type': match.group(2).strip()
                    })
                else:
                    # Fallback for symlinks without type info
                    symlinks.append({
                        'url': symlink_def,
                        'type': 'UNKNOWN'
                    })
        
        return symlinks
    
    def _parse_column_lineage(self, lineage_text: str) -> dict:
        """Parse the column lineage section"""
        lineage = {}
        
        if not lineage_text:
            return lineage
        
        # Split by lines and process each lineage entry
        lines = lineage_text.strip().split('\n')
        current_column = None
        
        for line in lines:
            line = line.strip()
            
            # Check if this is a new column definition (e.g., "product_type <-")
            if '<-' in line and not line.startswith(' '):
                column_match = re.match(r'(\w+)\s*<-', line)
                if column_match:
                    current_column = column_match.group(1)
                    lineage[current_column] = []
            
            # Check if this is a source column with transformation
            elif current_column and line.startswith('gs://'):
                # Parse source column path
                source_match = re.match(r'(gs://[^\s]+)', line)
                if source_match:
                    source_path = source_match.group(1)
                    
                    # Look for transformation type on the next line or same line
                    transform_type = "UNKNOWN"
                    if "Transform:" in line:
                        transform_match = re.search(r'Transform:\s*(.+)', line)
                        if transform_match:
                            transform_type = transform_match.group(1).strip()
                    
                    lineage[current_column].append({
                        'source_column': source_path,
                        'transform_type': transform_type
                    })
            
            # Check if this is a transformation line
            elif current_column and line.startswith('Transform:'):
                transform_match = re.search(r'Transform:\s*(.+)', line)
                if transform_match and lineage[current_column]:
                    # Update the last source column's transform type
                    lineage[current_column][-1]['transform_type'] = transform_match.group(1).strip()
        
        return lineage