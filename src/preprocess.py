import pandas as pd
import yaml
from pathlib import Path

# Load paramaters from params.yaml

params = yaml.safe_load(open("params.yaml"))['preprocess']

def preprocess(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Read data
    data = pd.read_csv(input_path)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save processed data
    data.to_csv(output_path, header=None, index=False)

    print(f"Preprocessed data saved to {output_path}")


if __name__ == "__main__":
    preprocess(
        input_path=params["input"],
        output_path=params["output"]
    )