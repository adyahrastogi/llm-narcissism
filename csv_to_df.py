import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

def preprocess(file_path):
    df = pd.read_csv(file_path)

    cols = [
        'human_answers',
        'chatgpt_answers',
        'deepseek-v2_answers',
        'llama-3.2_answers',
        'gemma3-12b_answers'
    ]

    df = df.dropna(subset=cols) # make sure none of the cells are nan

    samples = []
    for _, row in df.iterrows():
        question = row['question']
        
        # Human answers (label 0)
        samples.append({
            'text': f"Question: {question}\nResponse: {row['human_answers']}",
            'label': 0
        })
        
        # Llama-3.2 answers (label 1)
        samples.append({
            'text': f"Question: {question}\nResponse: {row['llama-3.2_answers']}",
            'label': 1
        })
        
        # Other LLM answers (label 2) - includes all non-Llama responses
        for llm_col in ['chatgpt_answers', 'deepseek-v2_answers', 'gemma3-12b_answers']:
            samples.append({
                'text': f"Question: {question}\nResponse: {row[llm_col]}",
                'label': 2
            })
    
    # Split into train/validation
    train_df, val_df = train_test_split(pd.DataFrame(samples), test_size=0.2, random_state=42)
    
    return Dataset.from_pandas(train_df), Dataset.from_pandas(val_df)

# Usage
train_dataset, eval_dataset = preprocess("llm_responses_checkpoint_branch2.csv")
