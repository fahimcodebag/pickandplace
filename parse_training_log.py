#!/usr/bin/env python3
"""Extract training data from terminal output text file and save as CSV.

Usage:
    python parse_training_log.py <input_text_file> [output_csv_file]

Examples:
    python parse_training_log.py training_log.txt
    python parse_training_log.py training_log.txt results.csv

The input file should contain lines in the format:
    Episode  1234 | Score:  123.456 | Avg:  100.000 | Best:  200.000 | Buf: 4,000,000
    Episode  1234 | ★ BEST! Score:  123.456 | Avg:  100.000 | Steps: 300 | Noise: 0.100

Non-matching lines (checkpoints, blank lines, etc.) are skipped.
"""

import re
import csv
import sys
import os


def parse_log(input_path, output_path=None):
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = base + '.csv'

    # Patterns for the two log line formats
    # Format 1: Episode  1234 | Score:  123.456 | Avg:  100.000 | Best:  200.000 | Buf: 4,000,000
    # Format 2: Episode  1234 | ★ BEST! Score:  123.456 | Avg:  100.000 | Steps: 300 | Noise: 0.100
    pattern_normal = re.compile(
        r'Episode\s+(\d+)\s*\|\s*Score:\s*([\d.\-]+)\s*\|\s*Avg:\s*([\d.\-]+)\s*\|\s*Best:\s*([\d.\-]+)\s*\|\s*Buf:\s*([\d,]+)'
    )
    pattern_best = re.compile(
        r'Episode\s+(\d+)\s*\|.*BEST.*Score:\s*([\d.\-]+)\s*\|\s*Avg:\s*([\d.\-]+)\s*\|\s*Steps:\s*(\d+)\s*\|\s*Noise:\s*([\d.\-]+)'
    )

    rows = []

    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()

            m = pattern_normal.match(line)
            if m:
                episode = int(m.group(1))
                score = float(m.group(2))
                avg = float(m.group(3))
                best = float(m.group(4))
                buf = int(m.group(5).replace(',', ''))
                rows.append({
                    'episode': episode,
                    'score': score,
                    'avg_100': avg,
                    'best': best,
                    'buffer': buf,
                    'is_best': False,
                    'steps': '',
                    'noise': '',
                })
                continue

            m = pattern_best.match(line)
            if m:
                episode = int(m.group(1))
                score = float(m.group(2))
                avg = float(m.group(3))
                steps = int(m.group(4))
                noise = float(m.group(5))
                rows.append({
                    'episode': episode,
                    'score': score,
                    'avg_100': avg,
                    'best': score,  # it IS the best
                    'buffer': '',
                    'is_best': True,
                    'steps': steps,
                    'noise': noise,
                })

    # Sort by episode number
    rows.sort(key=lambda r: r['episode'])

    # Write CSV
    fieldnames = ['episode', 'score', 'avg_100', 'best', 'buffer',
                  'is_best', 'steps', 'noise']
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Parsed {len(rows)} data points from: {input_path}")
    print(f"Saved to: {output_path}")

    # Print summary
    if rows:
        scores = [r['score'] for r in rows]
        avgs = [r['avg_100'] for r in rows]
        print(f"\nSummary:")
        print(f"  Episodes:    {rows[0]['episode']} → {rows[-1]['episode']}")
        print(f"  Score range: {min(scores):.3f} → {max(scores):.3f}")
        print(f"  Avg range:   {min(avgs):.3f} → {max(avgs):.3f}")
        print(f"  Best score:  {max(scores):.3f}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <input_text_file> [output_csv_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    parse_log(input_file, output_file)
