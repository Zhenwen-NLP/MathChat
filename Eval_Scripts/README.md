# README

This repository contains all the scripts needed for evaluation. Each script corresponds to a specific task and requires three input parameters:

1. **data_path**: This input should point to the source data file from MathChat.
2. **model_path**: This input should point to a large language model. We use vLLM for inference, so the tested models must be supported by vLLM.
3. **output_path**: This input specifies the path for the output result file.

Below is a brief overview of each script and its function:

## Script Usage

**Example Usage**: `python script.py --data_path <path_to_data_file> --model_path <path_to_model> --output_path <path_to_output_file>`

Ensure you provide the correct paths for each input parameter to successfully run the evaluation scripts. For any issues or questions, please open an issue or submit a pull request. Happy evaluating!