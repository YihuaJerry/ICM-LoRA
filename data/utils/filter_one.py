import json
import re
import os
import argparse

def filter_annotations(input_path, output_path, category):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not exist.")
        return

    # Define pattern to match only the specified category
    pattern = rf"{category}<loc_\d+><loc_\d+><loc_\d+><loc_\d+>"
    filtered_annotations = []

    # Read and process the file
    with open(input_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            
            # Extract matching category annotations
            matches = re.findall(pattern, data.get("suffix", ""))
            if matches:
                # Modify the "suffix" field to keep only matching annotations
                data["suffix"] = ''.join(matches)
                filtered_annotations.append(data)

    # Save the processed results
    with open(output_path, 'w') as output_file:
        for annotation in filtered_annotations:
            output_file.write(json.dumps(annotation) + '\n')

    print(f"Extracted {len(filtered_annotations)} annotations for category '{category}' to '{output_path}'")

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Filter annotations based on category')
    
    # Add command line arguments
    parser.add_argument('--category', type=str, required=True, help='Category to filter (e.g., sheep, dog, cat)')
    parser.add_argument('--train_input', type=str, default='./data/voc2012/train/annotations.jsonl', 
                        help='Input path for training annotations')
    parser.add_argument('--valid_input', type=str, default='./data/voc2012/valid/annotations.jsonl', 
                        help='Input path for validation annotations')
    parser.add_argument('--train_output', type=str, default=None, 
                        help='Output path for filtered training annotations')
    parser.add_argument('--valid_output', type=str, default=None, 
                        help='Output path for filtered validation annotations')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for filtered annotations')
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # If output paths are not specified, generate them based on input paths and category name
    if args.output_dir:
        train_output = os.path.join(args.output_dir, f'annotations_{args.category}.jsonl')
        valid_output = os.path.join(args.output_dir, f'annotations_{args.category}.jsonl')
    else:
        if not args.train_output:
            train_dir = os.path.dirname(args.train_input)
            args.train_output = os.path.join(train_dir, f'annotations_{args.category}.jsonl')
        
        if not args.valid_output:
            valid_dir = os.path.dirname(args.valid_input)
            args.valid_output = os.path.join(valid_dir, f'annotations_{args.category}.jsonl')
    
    # Process training set
    filter_annotations(
        input_path=args.train_input,
        output_path=args.train_output,
        category=args.category
    )
    
    # Process validation set
    filter_annotations(
        input_path=args.valid_input,
        output_path=args.valid_output,
        category=args.category
    )

if __name__ == "__main__":
    main()