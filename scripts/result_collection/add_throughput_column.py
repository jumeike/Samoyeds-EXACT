#!/usr/bin/env python3
"""
Post-process throughput CSV to add tokens_per_sec column.
Formula: tokens_per_sec = (batch_size * seq_len * 1000) / time_ms
"""

import sys
import pandas as pd

def add_throughput_column(csv_file):
    """Add tokens_per_sec column to the CSV file."""
    try:
        # Read the CSV
        df = pd.read_csv(csv_file)
        
        # Calculate tokens per second
        # tokens_per_sec = (batch_size × seq_len × 1000) / time_ms
        df['tokens_per_sec'] = (df['batch_size'] * df['seq_len'] * 1000) / df['time']
        
        # Round to 2 decimal places for readability
        df['tokens_per_sec'] = df['tokens_per_sec'].round(2)
        
        # Write back to the same file
        df.to_csv(csv_file, index=False)
        
        print(f"✓ Added tokens_per_sec column to {csv_file}")
        print(f"  Total rows: {len(df)}")
        print(f"  Throughput range: {df['tokens_per_sec'].min():.2f} - {df['tokens_per_sec'].max():.2f} tokens/sec")
        
        return True
        
    except Exception as e:
        print(f"✗ Error processing {csv_file}: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: add_throughput_column.py <csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    success = add_throughput_column(csv_file)
    sys.exit(0 if success else 1)
