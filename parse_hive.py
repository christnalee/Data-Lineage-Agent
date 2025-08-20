import re
import json
import os
import glob

def extract_sql_from_lineage_logs(input_log_path: str, output_file_path: str):
    """
    Parses a log file, extracts SQL query text and vertices from lines containing
    'hooks.LineageLogger' entries, and saves them to an output file.

    Args:
        input_log_path (str): Path to the input log file (e.g., 'hive.log').
        output_file_path (str): Path to the output text file where extracted
                                SQL queries and vertices will be stored.
    """
    extracted_entries = []
    
    # Regex to capture the JSON object directly following "hooks.LineageLogger:"
    lineage_logger_pattern = re.compile(r"hooks\.LineageLogger:\s*(\{.+?\})\s*$")

    print(f"Starting SQL and vertices extraction from: '{input_log_path}'")
    
    # Check if the input file actually exists before trying to open it
    if not os.path.exists(input_log_path):
        print(f"Error: Input log file '{input_log_path}' not found. Please ensure it exists and the path is correct.")
        return []

    try:
        with open(input_log_path, 'r', encoding='utf-8', errors='ignore') as infile:
            for line_num, line in enumerate(infile, 1):
                # Optimize by quickly checking if the identifying string is present
                if "hooks.LineageLogger:" in line:
                    match = lineage_logger_pattern.search(line)
                    if match:
                        json_str = match.group(1) # This captures the full JSON string
                        try:
                            data = json.loads(json_str)

                            entry = {}
                            
                            # Extract query text
                            if "queryText" in data and isinstance(data["queryText"], str):
                                entry["queryText"] = data["queryText"].strip()
                            
                            # Extract vertices
                            if "vertices" in data and isinstance(data["vertices"], list):
                                entry["vertices"] = data["vertices"]
                            
                            # Add source file information
                            entry["source_file"] = os.path.basename(input_log_path)
                            entry["line_number"] = line_num
                            
                            # Only add entry if we have at least query text or vertices
                            if entry.get("queryText") or entry.get("vertices"):
                                extracted_entries.append(entry)
                            else:
                                print(f"Warning: Neither 'queryText' nor 'vertices' found in JSON on line {line_num}. "
                                      f"Skipping this log entry. Log snippet (first 100 chars of JSON): {json_str[:100]}...")
                                
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON on line {line_num}: {e}")
                            print(f"Problematic JSON string (first 200 chars): {json_str[:200]}...")
                        except Exception as e:
                            print(f"An unexpected error occurred processing line {line_num}: {e}")
                            print(f"Line content (first 200 chars): {line.strip()[:200]}...")
                    else:
                        print(f"Warning: 'hooks.LineageLogger:' found on line {line_num}, "
                              f"but couldn't extract a complete JSON object using the regex. Line: {line.strip()}")
    except Exception as e:
        print(f"An error occurred while reading the input file '{input_log_path}': {e}")
        return []

    print(f"Found {len(extracted_entries)} entries with SQL queries and/or vertices from '{input_log_path}'")
    return extracted_entries


def extract_sql_from_all_hive_logs(hive_logs_folder: str, output_file_path: str):
    """
    Processes all .log files in the hive_logs folder and combines extracted data into one output file.
    
    Args:
        hive_logs_folder (str): Path to the folder containing hive log files
        output_file_path (str): Path to the output text file where combined extracted data will be stored
    """
    all_extracted_entries = []
    
    # Check if the hive_logs folder exists
    if not os.path.exists(hive_logs_folder):
        print(f"Error: Hive logs folder '{hive_logs_folder}' not found.")
        return
    
    # Find all .log files in the hive_logs folder
    log_files = glob.glob(os.path.join(hive_logs_folder, "*.log"))
    
    if not log_files:
        print(f"No .log files found in '{hive_logs_folder}'")
        return
    
    print(f"Found {len(log_files)} log files to process in '{hive_logs_folder}'")
    
    # Process each log file
    for log_file in sorted(log_files):
        print(f"\nProcessing: {os.path.basename(log_file)}")
        entries = extract_sql_from_lineage_logs(log_file, "")  # Pass empty string for output since we're combining
        all_extracted_entries.extend(entries)
    
    print(f"\nTotal entries extracted from all files: {len(all_extracted_entries)}")
    
    if not all_extracted_entries:
        print("No SQL queries or vertices from 'hooks.LineageLogger' entries were found in any log files.")
        return
    
    # Write combined results to output file
    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            
            for i, entry in enumerate(all_extracted_entries):
                outfile.write(f"--- Extracted Entry {i+1} ---\n")
                outfile.write(f"Source File: {entry.get('source_file', 'Unknown')}\n")
                outfile.write(f"Line Number: {entry.get('line_number', 'Unknown')}\n\n")
                
                # Write query text if available
                if entry.get("queryText"):
                    outfile.write("QUERY:\n")
                    outfile.write(entry["queryText"])
                    outfile.write("\n\n")
                
                # Write vertices if available
                if entry.get("vertices"):
                    outfile.write("VERTICES:\n")
                    # Pretty print the vertices JSON for readability
                    vertices_str = json.dumps(entry["vertices"], indent=2)
                    outfile.write(vertices_str)
                    outfile.write("\n\n")
                
                # Add separator between entries
                outfile.write("-" * 50 + "\n\n")
        
        print(f"Successfully extracted {len(all_extracted_entries)} entries to '{output_file_path}'.")
        
        # Print summary by file
        file_counts = {}
        for entry in all_extracted_entries:
            source_file = entry.get('source_file', 'Unknown')
            file_counts[source_file] = file_counts.get(source_file, 0) + 1
        
        print("\nEntries per file:")
        for filename, count in sorted(file_counts.items()):
            print(f"  {filename}: {count} entries")
            
    except Exception as e:
        print(f"Error writing to output file '{output_file_path}': {e}")


def extract_structured_data_for_rag(input_log_path: str, output_json_path: str):
    """
    Alternative function that extracts data in a more structured format optimized for RAG processing.
    Creates a JSON file with queries, vertices, and extracted table/column information.
    """
    extracted_data = []
    lineage_logger_pattern = re.compile(r"hooks\.LineageLogger:\s*(\{.+?\})\s*$")

    print(f"Starting structured extraction from: '{input_log_path}'")
    
    if not os.path.exists(input_log_path):
        print(f"Error: Input log file '{input_log_path}' not found.")
        return

    try:
        with open(input_log_path, 'r', encoding='utf-8', errors='ignore') as infile:
            for line_num, line in enumerate(infile, 1):
                if "hooks.LineageLogger:" in line:
                    match = lineage_logger_pattern.search(line)
                    if match:
                        json_str = match.group(1)
                        try:
                            data = json.loads(json_str)
                            
                            # Extract structured information
                            structured_entry = {
                                "entry_id": line_num,
                                "queryText": data.get("queryText", "").strip(),
                                "vertices": data.get("vertices", []),
                                "tables": [],
                                "columns": []
                            }
                            
                            # Process vertices to extract table and column information
                            if structured_entry["vertices"]:
                                for vertex in structured_entry["vertices"]:
                                    if isinstance(vertex, dict):
                                        vertex_type = vertex.get("vertexType", "")
                                        vertex_id = vertex.get("vertexId", "")
                                        
                                        if vertex_type == "TABLE":
                                            # Extract table information
                                            table_parts = vertex_id.split(".")
                                            database = table_parts[0] if len(table_parts) > 1 else ""
                                            table_name = table_parts[-1] if table_parts else vertex_id
                                            
                                            structured_entry["tables"].append({
                                                "vertex_id": vertex.get("id"),
                                                "database": database,
                                                "table_name": table_name,
                                                "full_name": vertex_id
                                            })
                                        
                                        elif vertex_type == "COLUMN":
                                            # Extract column information
                                            if "." in vertex_id:
                                                parts = vertex_id.split(".")
                                                if len(parts) >= 3:  # database.table.column
                                                    database = parts[0]
                                                    table_name = parts[1]
                                                    column_name = parts[2]
                                                elif len(parts) == 2:  # table.column
                                                    database = ""
                                                    table_name = parts[0]
                                                    column_name = parts[1]
                                                else:
                                                    database = ""
                                                    table_name = ""
                                                    column_name = vertex_id
                                            else:
                                                database = ""
                                                table_name = ""
                                                column_name = vertex_id
                                            
                                            structured_entry["columns"].append({
                                                "vertex_id": vertex.get("id"),
                                                "database": database,
                                                "table_name": table_name,
                                                "column_name": column_name,
                                                "full_name": vertex_id
                                            })
                            
                            if structured_entry["queryText"] or structured_entry["vertices"]:
                                extracted_data.append(structured_entry)
                                
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON on line {line_num}: {e}")
                        except Exception as e:
                            print(f"Error processing line {line_num}: {e}")
    
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Save structured data as JSON
    try:
        with open(output_json_path, 'w', encoding='utf-8') as outfile:
            json.dump(extracted_data, outfile, indent=2, ensure_ascii=False)
        print(f"Successfully extracted {len(extracted_data)} structured entries to '{output_json_path}'.")
    except Exception as e:
        print(f"Error writing structured data: {e}")


# --- Main execution block ---
if __name__ == "__main__":
    # Define your hive logs folder and the desired output file path
    hive_logs_folder = "hive_logs"
    output_queries_file = "extracted_hive.txt"

    # Process all hive logs and combine results
    extract_sql_from_all_hive_logs(hive_logs_folder, output_queries_file)