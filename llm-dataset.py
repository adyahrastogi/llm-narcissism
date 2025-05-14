from datasets import load_dataset
import pandas as pd
import ollama
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
import tqdm

logging.basicConfig(filename='llm-dataset-errors.log', level=logging.ERROR)

CHECKPOINT_FILE = "llm_responses_checkpoint.csv"
MAX_RETRIES = 3
RETRY_DELAY = 5
NUM_THREADS = 3
MODELS = ["deepseek-v2:latest", "llama3.2:latest", "gemma3:12b"]
MODEL_NAMES = ["deepseek-v2", "llama-3.2", "gemma3-12b"]

num_questions = 2000

def load_data():
    try:
        df = pd.read_csv(CHECKPOINT_FILE)
        print(f"Loaded {len(df)} responses from checkpoint file")

        # If we already have enough rows, return as-is
        if len(df) >= num_questions:
            return df.head(num_questions)

        # Otherwise, load more from the full dataset
        full_df = pd.read_csv("shuffled_hc3_dataset.csv")
        missing_rows = num_questions - len(df)
        extra_df = full_df.iloc[len(df):len(df) + missing_rows].copy()

        for model in MODEL_NAMES:
            col_name = f"{model}_answers"
            extra_df[col_name] = pd.NA

        # Append the new rows and return
        df = pd.concat([df, extra_df], ignore_index=True)
        print(f"Appended {missing_rows} more rows from shuffled dataset.")
        return df

    except FileNotFoundError:
        # Starting from scratch
        full_df = pd.read_csv("shuffled_hc3_dataset.csv")
        x_portion_of_df = full_df.head(num_questions).copy()

        for model in MODEL_NAMES:
            col_name = f"{model}_answers"
            x_portion_of_df[col_name] = pd.NA

        return x_portion_of_df

def get_llm_response(question, actual_model_name):
    for attempt in range(MAX_RETRIES):
        try:
            response = ollama.generate(
                model=actual_model_name,
                prompt=question,
                options={"temperature": 0.7, "top_p": 0.9, "num_predict": 250}
            )
            print(f"Completed response for model: {actual_model_name}")
            return response['response']
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                logging.error(f"Error generating response for question '{question}' with model '{actual_model_name}': {e}")
                return None
            
def process_row(row, model_names):
    if all(pd.notna(row[f"{model}_answers"]) for model in model_names):
        # print(f"Row {row['question']} already processed for models {model_names}")
        return row
    
    row_copy = row.copy()
    # print(f"Processing row {row['question']} for models {model_names}")
    futures = {}

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        for i in range(len(model_names)):
            col_name = f"{MODEL_NAMES[i]}_answers"
            # print(f"Checking {col_name} for question '{row['question']}: {row_copy[col_name]}', pd.isna: {pd.isna(row_copy[col_name])}")
            if pd.isna(row_copy[col_name]):
                future = executor.submit(get_llm_response, row_copy['question'], MODELS[i])
                # print(f"Submitting {col_name} for question '{row['question']}'")
                futures[future] = col_name
        for future in as_completed(futures):
            col_name = futures[future]
            try:
                response = future.result()
                row_copy[col_name] = response if response else pd.NA
            except Exception as e:
                logging.error(f"Error processing {col_name} for question '{row['question']}': {e}")
                row_copy[col_name] = pd.NA

    return row_copy


def main():
    df = load_data()
    print(df.head())

    # with tqdm(total=len(df), desc="Processing rows") as pbar:
    for idx in range(len(df)):
        # print(f"Starting to process {len(df)} rows")

        if all(pd.notna(df.loc[idx, f"{model}_answers"]) for model in MODEL_NAMES):
            # pbar.update(1)
            continue
        df.loc[idx] = process_row(df.loc[idx], MODEL_NAMES)

        if (idx + 1) % 5 == 0:
            df.to_csv(CHECKPOINT_FILE, index=False)
        if (idx + 1) % 5 == 0:
            print(f"Processed {idx + 1} rows out of {len(df)}")
        
        # pbar.update(1)
    df.to_csv(CHECKPOINT_FILE, index=False)
    print("Completed generating responses for specified number of questions.")

if __name__ == "__main__":
    try:
        available_models = [m['model'] for m in ollama.list()['models']]
        print("Available models: ", available_models)

        main() if len(available_models) == 3 else print("Please ensure you have the required models installed.")
        
    except Exception as e:
        print("Ollama is not running. Please start it and try again.")
        exit(1)
