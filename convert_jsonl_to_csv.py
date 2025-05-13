import json
import csv

# Define the input and output file paths
input_file = 'dataset/finance.jsonl'
output_file = 'dataset/finance.csv'

# Open the JSONL file and the CSV file
with open(input_file, 'r', encoding='utf-8') as jsonl_file, open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
    # Create a CSV writer
    csv_writer = csv.writer(csv_file)

    # Write the header to the CSV file
    headers_written = False

    # Process each line in the JSONL file
    for line in jsonl_file:
        # Parse the JSON object
        json_obj = json.loads(line)

        # If the header hasn't been written yet, write it
        if not headers_written:
            headers = json_obj.keys()
            csv_writer.writerow(headers)
            headers_written = True

        # Write the JSON object values to the CSV file
        csv_writer.writerow(json_obj.values())

print(f'Conversion complete. CSV file saved as {output_file}.')
