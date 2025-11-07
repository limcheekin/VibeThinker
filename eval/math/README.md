# Math Evaluation Guide

This document provides a ready-to-use math evaluation program for the VibeThinker-1.5B model.

## Evaluation Process

#### 1. Clone the Required Project

First, you need to clone the `deepcoder` branch of the `rllm` project.

```bash
git clone https://github.com/rllm-org/rllm.git
cd rllm
git checkout deepcoder
```

#### 2. Place the Evaluation File

Move the provided `main_evaluation.py` file into the `verl/verl/trainer/` directory of the `rllm` project.

#### 3. Install Dependencies

Navigate to the `rllm` directory and install the required environment and dependencies according to its official guide.

```bash
# Example installation command. Please refer to the deepcoder project for specific requirements.
pip install -e ./verl
pip install -e .
```

#### 4. Configure the Evaluation Script

Modify the following two variables in the `eval_model.sh` script:

-   `OUTPUT_DIR`: Specify the output directory for the evaluation results.
-   `DATA_PATH`: Specify the path to the evaluation dataset. We have provided the processed dataset in the `data` folder; please point this variable to its location.

For example:
```bash
# eval_model.sh

OUTPUT_DIR="./eval_output"
DATA_PATH="../data" # Assuming the data folder is in the parent directory of the rllm project
```

#### 5. Run the Evaluation

After completing the configuration, execute the evaluation script:

```bash
bash eval_model.sh
```

## Key Features

-   **Multi-GPU and Multi-Node Support**: The evaluation code natively supports distributed evaluation across multiple GPUs and nodes to accelerate the process.
-   **Pre-processed Dataset**: We provide a pre-processed evaluation dataset in the `data` folder, allowing you to start the evaluation directly.

## Acknowledgements

This evaluation program is built upon the [rllm/deepcoder](https://github.com/rllm-org/rllm/tree/deepcoder) project. Thanks to the original authors for their contributions.
