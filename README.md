# Deep Learning Midterm Project: Math Question Answer Verification

## Authors

- **Tinghao (Gavin) Zhang** - tz2769
- **Haonan (Gary) Guo** - hg2924

## Project Background

This project aims to address the challenge of verifying answers to math problems. The core task is to fine-tune a large language model (Llama-3.1-8B) to accurately determine whether a given solution to a math problem is correct. The model's final output should be a simple boolean value: `True` (correct) or `False` (incorrect).

This project is part of the "Math Question Answer Verification Competition," and all experiments and code are centered around the goals of this competition.

## Dataset

This project utilizes a public dataset from Hugging Face:

- **Dataset Name:** `ad6398/nyu-dl-teach-maths-comp`
- **Content:** The dataset contains a large number of math questions, corresponding solutions, and a label (`is_correct`) indicating whether the solution is correct.

In the various notebooks, we have partitioned the full dataset into different sizes for training and validation sets based on our experimental needs.

## Tech Stack and Core Methods

To efficiently fine-tune a large model on limited computing resources (like the free tier of Google Colab), we employed the following key technologies:

- **Model:** `unsloth/Meta-Llama-3.1-8B`, the only model specified for this competition.
- **Efficient Fine-tuning Library:** `unsloth`, which makes it possible to fine-tune large models on a single GPU by optimizing memory usage and computation speed.
- **4-bit Quantization:** Compressing the model's weights from standard 32-bit or 16-bit floating-point numbers to 4-bit integers, which significantly reduces GPU memory consumption.
- **LoRA (Low-Rank Adaptation):** A parameter-efficient fine-tuning technique. It speeds up training and reduces resource usage by adding small, trainable "adapter" matrices alongside the model's existing layers, rather than training the entire model's billions of parameters.

## Notebook Descriptions

This project includes several Jupyter Notebook files that document the different stages and experimental iterations of our problem-solving process.

### `DL_mid01.ipynb`

- **Description:** This is the initial starter script for the project, designed to quickly run through the entire workflow.
- **Configuration:** Uses a smaller subset of the data (5,000 training samples and 500 validation samples) and basic training parameters.
- **Objective:** To serve as a baseline model and verify the correctness of all stages, including data loading, model training, and submission file generation.

### `DL_mid02.ipynb`

- **Description:** An extended experiment based on the first version.
- **Configuration:** Increased the amount of training data (10,000 training samples and 1,000 validation samples) and made initial adjustments to hyperparameters such as learning rate and batch size.
- **Objective:** To explore the impact of larger-scale data on model performance.

### `DL_mid03.ipynb` & `DL_midterm0_81.ipynb`

- **Description:** These two notebooks are the advanced versions at the core of the project, introducing more robust training and evaluation mechanisms.
- **Configuration:**
    - Used 10,000 training samples and 1,000 validation samples.
    - Adopted an epoch-based training approach to allow the model to learn more thoroughly.
    - Introduced validation set evaluation during training (`eval_strategy = "steps"`), enabling real-time monitoring of the model's performance on unseen data and saving the best model.
    - Implemented more detailed training logs and performance reports (achieving validation accuracies of **82.1%** and **81.8%**, respectively).
- **Objective:** To find a better-performing model through systematic evaluation and hyperparameter tuning.

### `LR1027-05.final.ipynb`

- **Description:** A highly integrated and compact final version script.
- **Configuration:**
    - Used a larger dataset (50,000 training samples and 5,000 validation samples).
    - Employed a different prompt format.
    - Consolidated all steps (from environment setup to submission file generation) into a single code cell for easy execution and reproducibility.
- **Objective:** To train on a larger scale and serve as a final, one-click runnable solution.

## Model Weights

The link to the trained model weights can be found in the `ModelWeight_Link.md` file.

## Final Submission

The final submission file submitted to Kaggle is `submission 0.83.csv`, which achieved a score of 0.83 on the competition leaderboard.
