# simulator.py

import os
import time
import csv
from datetime import datetime
import random

# --- Configuration ---
# This is the directory ON YOUR HOST machine that is mapped
# to /app/data/input inside the Pathway container.
# Make sure this path matches the first part of the -v volume mount
# in your 'docker run' command.
SIMULATOR_OUTPUT_DIR = "./local_input_data"
INTERVAL_SECONDS = 10  # How often to generate a new batch of tickets
BATCH_SIZE = 3       # How many tickets to generate in each batch

# --- Ensure the output directory exists ---
os.makedirs(SIMULATOR_OUTPUT_DIR, exist_ok=True)
print(f"Simulator configured to write files to: {os.path.abspath(SIMULATOR_OUTPUT_DIR)}")
print(f"Generating {BATCH_SIZE} tickets every {INTERVAL_SECONDS} seconds...")

# --- Main Simulation Loop ---
ticket_counter = 0
try:
    while True:
        print(f"\nSIMULATOR: Waiting {INTERVAL_SECONDS}s...")
        time.sleep(INTERVAL_SECONDS)

        # Generate a unique timestamp string for the filename and data
        # Using microseconds for higher chance of unique filenames if interval is very short
        timestamp_obj = datetime.now()
        timestamp_str_iso = timestamp_obj.isoformat()
        timestamp_str_file = timestamp_obj.strftime("%Y%m%d_%H%M%S_%f") # Format suitable for filenames
        filename = os.path.join(SIMULATOR_OUTPUT_DIR, f"tickets_{timestamp_str_file}.csv")

        print(f"SIMULATOR: Generating batch file: {filename}")

        # Generate ticket data for the batch
        batch_data = []
        for i in range(BATCH_SIZE):
            ticket_id = f"TKT-{ticket_counter:06d}"
            # Generate slightly more varied customer IDs
            customer_id = f"CUST-{(ticket_counter % 150):04d}"
            # Add some random variation to subjects/bodies
            issue_type = random.choice(["login", "payment", "profile update", "feature request", "bug report"])
            urgency = random.choice(["low", "medium", "high"])
            subject = f"Issue with {issue_type} - Urgency: {urgency} ({ticket_id})"
            body = (f"User {customer_id} reported an issue regarding {issue_type}. "
                    f"Timestamp: {timestamp_str_iso}. Details received: Error code E{random.randint(100,999)}. "
                    f"Please investigate ticket {ticket_id}.")

            batch_data.append([ticket_id, timestamp_str_iso, customer_id, subject, body])
            ticket_counter += 1

        # Write the batch to a new CSV file
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Write header - MUST match the schema in pathway_script.py
                writer.writerow(["ticket_id", "timestamp", "customer_id", "subject", "body"])
                # Write data rows
                writer.writerows(batch_data)
            print(f"SIMULATOR: Successfully created {filename} with {len(batch_data)} tickets.")
        except IOError as e:
            print(f"SIMULATOR: Error writing file {filename}: {e}")
        except Exception as e:
            print(f"SIMULATOR: An unexpected error occurred during file writing: {e}")


except KeyboardInterrupt:
    print("\nSIMULATOR: Simulation stopped by user (Ctrl+C).")
except Exception as e:
    print(f"\nSIMULATOR: An unexpected error occurred in the main loop: {e}")