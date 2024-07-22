import json
import pandas as pd

def decode_line(line):
    with open('../../models/results/associativeRules.txt', 'r') as f:
        file_content = f.read()
        file_content = file_content.replace(' = ', '=')

    lines = file_content.split('\n')
    decoded_lines = [decode_line(line) for line in lines]

    decoded_content = '\n'.join(decoded_lines)

    with open('label_mappings.json', 'r') as f:
        label_mappings = json.load(f)

    reverse_mappings = {}
    for attribute, mapping in label_mappings.items():
        reverse_mappings[attribute] = {v: k for k, v in mapping.items()}
        
    if ' THEN ' in line:
        rule_parts = line.split(' THEN ')
        antecedent = rule_parts[0].strip()
        consequent = rule_parts[1].strip()
        
        if '=' in antecedent:
            attribute, value = antecedent.split('=')
            attribute = attribute.strip()
            if attribute == 'priority':
                value = float(value)
            else:
                value = int(value)
            if attribute in reverse_mappings:
                decoded_value = reverse_mappings[attribute].get(value, f"Unknown_{value}")
                antecedent = f"{attribute}={decoded_value}"
        
        if '=' in consequent:
            attribute, value = consequent.split('=')
            attribute = attribute.strip()
            if attribute == 'priority':
                value = float(value)
            else:
                value = int(value.strip())
            if attribute in reverse_mappings:
                decoded_value = reverse_mappings[attribute].get(value, f"Unknown_{value}")
                consequent = f"{attribute}={decoded_value}"
    with open('../../models/results/decoded.txt', 'w') as f:
        f.write(decoded_content)

