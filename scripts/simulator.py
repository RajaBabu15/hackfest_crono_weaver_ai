# python scripts/simulator.py

import os
import time
import csv
from datetime import datetime
import random

SIMULATOR_OUTPUT_DIR = "./data/input" # Corrected path
INTERVAL_SECONDS = 15
BATCH_SIZE = 5

# Ensure the target directory exists relative to the script's CWD when run
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the absolute path for the output directory
absolute_output_dir = os.path.join(script_dir, '..', 'data', 'input') # Go up one level from 'scripts', then into 'data/input'

os.makedirs(absolute_output_dir, exist_ok=True)
print(f"Simulator writing files to: {absolute_output_dir}") # Print absolute path
print(f"Generating {BATCH_SIZE} tickets every {INTERVAL_SECONDS} seconds...")

ticket_counter = 0
try:
    while True:
        print(f"\nSIMULATOR: Waiting {INTERVAL_SECONDS}s...")
        time.sleep(INTERVAL_SECONDS)

        timestamp_obj = datetime.now()
        timestamp_str_iso = timestamp_obj.isoformat()
        timestamp_str_file = timestamp_obj.strftime("%Y%m%d%H%M%S_%f")
        # Use the absolute path for creating the filename
        filename = os.path.join(absolute_output_dir, f"tickets_{timestamp_str_file}.csv")

        print(f"SIMULATOR: Generating batch file: {filename}")

        batch_data = []
        for i in range(BATCH_SIZE):
            ticket_id = f"TKT-{ticket_counter:06d}"
            customer_id = f"CUST-{(ticket_counter % 150):04d}"
            issue_type = random.choice(["login", "payment", "profile update", "feature request", "bug report", "general inquiry", "password reset"])
            urgency = random.choice(["low", "medium", "high", "critical"])
            subject = f"Issue with {issue_type} - Urgency: {urgency} ({ticket_id})"
            body = (f"User {customer_id} reported an issue regarding {issue_type}. "
                    f"Timestamp: {timestamp_str_iso}. Details received: Error code E{random.randint(100,999)}. "
                    f"Please investigate ticket {ticket_id}. Related user agent: {random.choice(['Chrome', 'Firefox', 'Safari', 'MobileApp'])}. "
                    f"Follow up needed: {random.choice(['Yes', 'No'])}.")

            batch_data.append([ticket_id, timestamp_str_iso, customer_id, subject, body])
            ticket_counter += 1

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["ticket_id", "timestamp", "customer_id", "subject", "body"])
                writer.writerows(batch_data)
            print(f"SIMULATOR: Created {filename} with {len(batch_data)} tickets.")
        except IOError as e:
            print(f"SIMULATOR: Error writing file {filename}: {e}")
        except Exception as e:
            print(f"SIMULATOR: Unexpected error during file writing: {e}")

except KeyboardInterrupt:
    print("\nSIMULATOR: Simulation stopped.")
except Exception as e:
    print(f"\nSIMULATOR: Unexpected error in main loop: {e}")