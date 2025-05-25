from csv_to_df import preprocess
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import optuna

# 1) Prepare tokenizer and your self-classification dataset
#    Assume you have a DatasetDict: {'train': train_ds, 'eval': eval_ds}
#    where examples are {"text": ..., "label": 0 or 1}.
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2")
def tokenize_batch(batch):
    return tokenizer(batch["text"], truncation=True)

train_dataset, eval_dataset = preprocess("llm_responses_checkpoint_branch2.csv")

train_dataset = train_dataset.map(tokenize_batch, batch =True)
eval_dataset = eval_dataset.map(tokenize_batch, batch =True)

data_collator = DataCollatorWithPadding(tokenizer)

# 2) Wrap model initialization so that hyperparams can be sampled per trial
def model_init(trial=None):
    # Base classification model
    model = AutoModelForSequenceClassification.from_pretrained(
        "meta-llama/Llama-3.2",
        num_labels=2,
    )
    # Sample LoRA hyperparams if trial is provided
    r      = trial.suggest_int("lora_r",      4, 16) if trial else 8
    alpha  = trial.suggest_int("lora_alpha",  8, 32) if trial else 16
    dropout= trial.suggest_float("lora_dropout", 0.0, 0.3) if trial else 0.05

    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
    )
    return get_peft_model(model, peft_cfg)

# 3) Define compute_metrics
from sklearn.metrics import accuracy_score, f1_score
def compute_metrics(pred):
    labels = pred.label_ids
    preds  = np.argmax(pred.predictions, axis=-1)
    return {"accuracy": accuracy_score(labels, preds),
            "f1":       f1_score(labels, preds)}

# 4) Base training arguments
base_args = TrainingArguments(
    output_dir="./llama3_classify",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model_init     = model_init,
    args           = base_args,
    train_dataset  = train_dataset,
    eval_dataset   = eval_dataset,
    tokenizer      = tokenizer,
    data_collator = data_collator,
    compute_metrics= compute_metrics,
)

# 5) Define the hyperparameter search space
def hp_space_optuna(trial):
    return {
        # learning rate
        "learning_rate": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        # number of training epochs
        "num_train_epochs": trial.suggest_int("epochs", 1, 5),
        # per-device batch size
        "per_device_train_batch_size": trial.suggest_categorical("bs", [8, 16, 32]),
        # you can also sample warmup steps
        "warmup_steps": trial.suggest_int("warmup_steps", 0, 500),
        # LoRA hyperparams are sampled in model_init
    }

# 6) Run the search
best_run = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=hp_space_optuna,
    n_trials=20,
)

print("→ Best hyperparams:", best_run.hyperparameters)
# 7) (Optionally) re-train a final model with best_run.hyperparameters
trainer.args.update(best_run.hyperparameters)
final_model = trainer.train()
final_model.save_pretrained("./llama3_classify_best")
