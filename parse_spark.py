import json
import os
import glob

def parse_spark_events(input_file, source_filename):
    """
    Parse Spark OpenLineage events to extract table/column lineage information
    """
    
    with open(input_file, 'r') as f:
        content = f.read().strip()
    
    # Split by lines since each line is a separate JSON event
    lines = content.split('\n')
    
    extracted_entries = []
    
    for line_num, line in enumerate(lines, 1):
        if not line.strip():
            continue
            
        try:
            event = json.loads(line)
            
            # Only process events that have inputs/outputs (START, RUNNING, COMPLETE events)
            if 'inputs' not in event or 'outputs' not in event:
                continue
            
            # Skip events with empty inputs and outputs
            if not event['inputs'] and not event['outputs']:
                continue
            
            entry = {
                'entry_number': len(extracted_entries) + 1,
                'source_file': source_filename,
                'line_number': line_num,
                'event_type': event.get('eventType', 'UNKNOWN'),
                'event_time': event.get('eventTime', ''),
                'job_name': event.get('job', {}).get('name', ''),
                'run_id': event.get('run', {}).get('runId', ''),
                'sql_query': None,
                'inputs': [],
                'outputs': []
            }
            
            # Extract SQL query if available
            job_facets = event.get('job', {}).get('facets', {})
            if 'sql' in job_facets:
                entry['sql_query'] = job_facets['sql'].get('query', '')
            
            # Extract input datasets
            for input_dataset in event['inputs']:
                input_info = {
                    'namespace': input_dataset.get('namespace', ''),
                    'name': input_dataset.get('name', ''),
                    'full_name': f"{input_dataset.get('namespace', '')}/{input_dataset.get('name', '')}",
                    'columns': []
                }
                
                # Extract schema information
                if 'facets' in input_dataset and 'schema' in input_dataset['facets']:
                    schema = input_dataset['facets']['schema']
                    if 'fields' in schema:
                        for field in schema['fields']:
                            input_info['columns'].append({
                                'name': field.get('name', ''),
                                'type': field.get('type', '')
                            })
                
                # Extract symlinks (alternative names like Hive table names)
                if 'facets' in input_dataset and 'symlinks' in input_dataset['facets']:
                    symlinks = input_dataset['facets']['symlinks']
                    if 'identifiers' in symlinks:
                        input_info['symlinks'] = []
                        for symlink in symlinks['identifiers']:
                            input_info['symlinks'].append({
                                'namespace': symlink.get('namespace', ''),
                                'name': symlink.get('name', ''),
                                'type': symlink.get('type', '')
                            })
                
                entry['inputs'].append(input_info)
            
            # Extract output datasets
            for output_dataset in event['outputs']:
                output_info = {
                    'namespace': output_dataset.get('namespace', ''),
                    'name': output_dataset.get('name', ''),
                    'full_name': f"{output_dataset.get('namespace', '')}/{output_dataset.get('name', '')}",
                    'columns': []
                }
                
                # Extract schema information
                if 'facets' in output_dataset and 'schema' in output_dataset['facets']:
                    schema = output_dataset['facets']['schema']
                    if 'fields' in schema:
                        for field in schema['fields']:
                            output_info['columns'].append({
                                'name': field.get('name', ''),
                                'type': field.get('type', '')
                            })
                
                # Extract column lineage information
                if 'facets' in output_dataset and 'columnLineage' in output_dataset['facets']:
                    col_lineage = output_dataset['facets']['columnLineage']
                    output_info['column_lineage'] = {}
                    
                    if 'fields' in col_lineage:
                        for field_name, lineage_info in col_lineage['fields'].items():
                            output_info['column_lineage'][field_name] = []
                            if 'inputFields' in lineage_info:
                                for input_field in lineage_info['inputFields']:
                                    output_info['column_lineage'][field_name].append({
                                        'namespace': input_field.get('namespace', ''),
                                        'name': input_field.get('name', ''),
                                        'field': input_field.get('field', ''),
                                        'transformations': input_field.get('transformations', [])
                                    })
                
                entry['outputs'].append(output_info)
            
            extracted_entries.append(entry)
            
        except json.JSONDecodeError as e:
            print(f"Error parsing line {line_num} in {source_filename}: {e}")
            continue
        except Exception as e:
            print(f"Error processing line {line_num} in {source_filename}: {e}")
            continue
    
    return extracted_entries


def parse_all_spark_logs(spark_logs_folder, output_file):
    """
    Process all JSON files in the spark_logs folder and combine extracted data into one output file.
    
    Args:
        spark_logs_folder (str): Path to the folder containing spark log files
        output_file (str): Path to the output text file where combined extracted data will be stored
    """
    all_extracted_entries = []
    
    # Check if the spark_logs folder exists
    if not os.path.exists(spark_logs_folder):
        print(f"Error: Spark logs folder '{spark_logs_folder}' not found.")
        return
    
    # Find all JSON files in the spark_logs folder
    json_files = glob.glob(os.path.join(spark_logs_folder, "*.json"))
    
    if not json_files:
        print(f"No .json files found in '{spark_logs_folder}'")
        return
    
    print(f"Found {len(json_files)} JSON files to process in '{spark_logs_folder}'")
    
    # Process each JSON file
    for json_file in sorted(json_files):
        source_filename = os.path.basename(json_file)
        print(f"Processing: {source_filename}")
        entries = parse_spark_events(json_file, source_filename)
        
        # Renumber entries to be unique across all files
        for entry in entries:
            entry['entry_number'] = len(all_extracted_entries) + 1
            all_extracted_entries.append(entry)
    
    print(f"\nTotal entries extracted from all files: {len(all_extracted_entries)}")
    
    if not all_extracted_entries:
        print("No Spark events with inputs/outputs were found in any JSON files.")
        return
    
    # Write combined results to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # f.write(f"=== COMBINED SPARK LOG EXTRACTION ===\n")
            # f.write(f"Total files processed: {len(json_files)}\n")
            # f.write(f"Total entries extracted: {len(all_extracted_entries)}\n")
            # f.write(f"Generated on: {os.popen('date').read().strip()}\n\n")
            # f.write("=" * 80 + "\n\n")
            
            for entry in all_extracted_entries:
                f.write(f"--- Extracted Entry {entry['entry_number']} ---\n")
                f.write(f"SOURCE FILE: {entry['source_file']}\n")
                f.write(f"LINE NUMBER: {entry['line_number']}\n")
                f.write(f"EVENT TYPE: {entry['event_type']}\n")
                f.write(f"EVENT TIME: {entry['event_time']}\n")
                f.write(f"JOB NAME: {entry['job_name']}\n")
                f.write(f"RUN ID: {entry['run_id']}\n")
                
                if entry['sql_query']:
                    f.write(f"SQL QUERY:\n{entry['sql_query']}\n\n")
                
                # Write inputs
                f.write("INPUTS:\n")
                for i, input_dataset in enumerate(entry['inputs']):
                    f.write(f"  Input {i+1}:\n")
                    f.write(f"    Full Name: {input_dataset['full_name']}\n")
                    f.write(f"    Namespace: {input_dataset['namespace']}\n")
                    f.write(f"    Name: {input_dataset['name']}\n")
                    
                    if input_dataset['columns']:
                        f.write(f"    Columns:\n")
                        for col in input_dataset['columns']:
                            f.write(f"      - {col['name']} ({col['type']})\n")
                    
                    if 'symlinks' in input_dataset:
                        f.write(f"    Symlinks:\n")
                        for symlink in input_dataset['symlinks']:
                            f.write(f"      - {symlink['namespace']}/{symlink['name']} ({symlink['type']})\n")
                    f.write("\n")
                
                # Write outputs
                f.write("OUTPUTS:\n")
                for i, output_dataset in enumerate(entry['outputs']):
                    f.write(f"  Output {i+1}:\n")
                    f.write(f"    Full Name: {output_dataset['full_name']}\n")
                    f.write(f"    Namespace: {output_dataset['namespace']}\n")
                    f.write(f"    Name: {output_dataset['name']}\n")
                    
                    if output_dataset['columns']:
                        f.write(f"    Columns:\n")
                        for col in output_dataset['columns']:
                            f.write(f"      - {col['name']} ({col['type']})\n")
                    
                    if 'column_lineage' in output_dataset:
                        f.write(f"    Column Lineage:\n")
                        for col_name, lineage in output_dataset['column_lineage'].items():
                            f.write(f"      {col_name} <- \n")
                            for source in lineage:
                                f.write(f"        {source['namespace']}/{source['name']}.{source['field']}\n")
                                if source['transformations']:
                                    for transform in source['transformations']:
                                        f.write(f"          Transform: {transform.get('type', '')} - {transform.get('subtype', '')}\n")
                    f.write("\n")
                
                f.write("-" * 50 + "\n\n")
        
        print(f"Successfully extracted {len(all_extracted_entries)} entries to '{output_file}'.")
        
        # Print summary by file
        file_counts = {}
        for entry in all_extracted_entries:
            source_file = entry.get('source_file', 'Unknown')
            file_counts[source_file] = file_counts.get(source_file, 0) + 1
        
        print("\nEntries per file:")
        for filename, count in sorted(file_counts.items()):
            print(f"  {filename}: {count} entries")
            
    except Exception as e:
        print(f"Error writing to output file '{output_file}': {e}")


if __name__ == "__main__":
    # Define your spark logs folder and the desired output file path
    spark_logs_folder = "spark_logs"
    output_file = "extracted_spark.txt"
    
    # Process all spark logs and combine results
    parse_all_spark_logs(spark_logs_folder, output_file)