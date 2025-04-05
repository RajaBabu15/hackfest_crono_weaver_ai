# simulator_monitor.py
# This script runs independently to monitor the simulator status

import os
import time
import csv
from datetime import datetime
import streamlit as st

# Configuration 
INPUT_DIR = "./local_input_data"
OUTPUT_PATH = "./local_output_data/indexed_tickets.csv"

def count_tickets_in_file(file_path):
    """Count the number of tickets in a CSV file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Skip header
            next(csv.reader(f))
            return sum(1 for _ in csv.reader(f))
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    st.title("Simulator & Pipeline Monitor")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Simulator Status")
        simulator_status = st.empty()
    
    with col2:
        st.header("Indexing Status")
        indexing_status = st.empty()
    
    # Auto-refresh checkbox
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_rate = st.slider("Refresh rate (seconds)", min_value=1, max_value=30, value=5)
    
    # Manual refresh button
    refresh_button = st.button("Refresh Now")
    
    # Main monitoring loop
    while auto_refresh or refresh_button:
        refresh_button = False  # Reset the button state
        
        # Monitor simulator input directory
        try:
            input_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
            
            # Sort by modification time (newest first)
            input_files.sort(key=lambda x: os.path.getmtime(os.path.join(INPUT_DIR, x)), reverse=True)
            
            # Display simulator stats
            with simulator_status.container():
                st.write(f"Total input files: {len(input_files)}")
                
                if len(input_files) > 0:
                    st.write("Recent files:")
                    for i, file in enumerate(input_files[:5]):  # Show last 5 files
                        file_path = os.path.join(INPUT_DIR, file)
                        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        ticket_count = count_tickets_in_file(file_path)
                        
                        st.info(
                            f"📄 {file}\n"
                            f"Created: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                            f"Tickets: {ticket_count}"
                        )
                else:
                    st.warning("No input files found. Simulator may not be running.")
        except Exception as e:
            with simulator_status.container():
                st.error(f"Error monitoring simulator: {str(e)}")
        
        # Monitor output index file
        try:
            if os.path.exists(OUTPUT_PATH):
                # Get file stats
                mod_time = datetime.fromtimestamp(os.path.getmtime(OUTPUT_PATH))
                file_size = os.path.getsize(OUTPUT_PATH)
                ticket_count = count_tickets_in_file(OUTPUT_PATH)
                
                # Display indexing stats
                with indexing_status.container():
                    st.write(f"Index file: {os.path.basename(OUTPUT_PATH)}")
                    st.write(f"Last updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"File size: {file_size/1024:.2f} KB")
                    st.write(f"Indexed tickets: {ticket_count}")
                    
                    # Calculate processing rate if possible
                    if isinstance(ticket_count, int) and ticket_count > 0:
                        st.success("✅ Pipeline is working!")
                    else:
                        st.warning("⚠️ Pipeline may be having issues")
            else:
                with indexing_status.container():
                    st.warning(f"Output file does not exist: {OUTPUT_PATH}")
                    st.write("The pathway_script.py may not be running yet")
        except Exception as e:
            with indexing_status.container():
                st.error(f"Error monitoring indexing: {str(e)}")
        
        # Wait before next refresh
        if auto_refresh:
            time.sleep(refresh_rate)
            st.experimental_rerun()
        else:
            break

if __name__ == "__main__":
    main()