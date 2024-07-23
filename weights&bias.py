import wandb
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
from pytorch_tabular.models import AutoIntConfig


import wandb
wandb.login()

# Define the sweep configuration
sweep_config = {
    'method': 'random',  # Can be 'grid', 'random', 'bayes'
    'metric': {
        'name': 'valid_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'batch_size': {
            'values': [256, 512, 1024, 2048]
        },
        'max_epochs': {
            'values': [50, 200]
        },
        'min_epochs': {
            'values': [1, 50]
        },
        'early_stopping': {
            'values': [None, 'valid_loss']
        },
        'early_stopping_min_delta': {
            'values': [0.0001, 0.001]
        },
        'early_stopping_mode': {
            'values': ['min', 'max']
        },
        'early_stopping_patience': {
            'values': [3, 10]
        },
        'gradient_clip_val': {
            'values': [0.0, 1.0]
        },
        'auto_lr_find': {
            'values': [True, False]
        },
        'optimizer': {
            'values': ['Adam', 'SGD', 'RMSprop', 'AdamW']
        },
        'optimizer_params': {
            'values': [{'weight_decay': 0.0}, {'weight_decay': 0.1}]
        },
        'lr_scheduler': {
            'values': ['CosineAnnealingLR', 'StepLR', 'ReduceLROnPlateau']
        },
        'cosine_annealing_lr_params': {
            'values': [{'T_max': 10, 'eta_min': 0}]
        },
        'step_lr_params': {
            'values': [
                {'step_size': 5, 'gamma': 0.1},
                {'step_size': 5, 'gamma': 0.5},
                {'step_size': 10, 'gamma': 0.1},
                {'step_size': 10, 'gamma': 0.5}
            ]
        },
        'reduce_lr_on_plateau_params': {
            'values': [
                {'factor': 0.1, 'patience': 10},
                {'factor': 0.5, 'patience': 5}
            ]
        },
        'lr_scheduler_monitor_metric': {
            'values': ['valid_loss']
        },
        'dropout': {
            'values': [0.1, 0.5]
        },
        'learning_rate': {
            'values': [1e-5, 1e-2]
        }
    }
}

# Initialize a sweep
sweep_id = wandb.sweep(sweep_config, project='your_project_name')

# Function to train your model
def train():
    # Initialize a new run
    wandb.init()

    # Fetch hyperparameters
    config = wandb.config

    # Create the model configuration with fetched parameters
    data_config = DataConfig(
        target=["is_fraud"],
        continuous_cols=num_col_names,
        categorical_cols=cat_col_names
    )

    trainer_config = TrainerConfig(
        batch_size=config.batch_size,
        max_epochs=config.max_epochs,
        min_epochs=config.min_epochs,
        early_stopping=config.early_stopping,
        early_stopping_min_delta=config.early_stopping_min_delta,
        early_stopping_mode=config.early_stopping_mode,
        early_stopping_patience=config.early_stopping_patience,
        gradient_clip_val=config.gradient_clip_val,
        auto_lr_find=config.auto_lr_find,
    )

    optimizer_config = OptimizerConfig(
        optimizer=config.optimizer,
        optimizer_params=config.optimizer_params,
        lr_scheduler=config.lr_scheduler,
        lr_scheduler_monitor_metric=config.lr_scheduler_monitor_metric,
    )

    if config.lr_scheduler == 'CosineAnnealingLR':
        optimizer_config.lr_scheduler_params = config.cosine_annealing_lr_params
    elif config.lr_scheduler == 'StepLR':
        optimizer_config.lr_scheduler_params = config.step_lr_params
    elif config.lr_scheduler == 'ReduceLROnPlateau':
        optimizer_config.lr_scheduler_params = config.reduce_lr_on_plateau_params

    model_config = AutoIntConfig(
        task="classification",
        dropout=config.dropout,
        learning_rate=config.learning_rate,
    )

    tabular_model = TabularModel(
        data_config=data_config,
        trainer_config=trainer_config,
        optimizer_config=optimizer_config,
        model_config=model_config,
        verbose=True
    )

    # Fit the model with the training and validation data
    tabular_model.fit(train=train_data, validation=val_data)

    # Evaluate the model
    results = tabular_model.evaluate(test=test_data)

    # Convert the results to a dictionary if needed
    if isinstance(results, list):
        results_dict = {f"metric_{i}": result for i, result in enumerate(results)}
    else:
        results_dict = results

    # Log metrics (replace with your evaluation logic)
    wandb.log(results_dict)

# Run the sweep
wandb.agent(sweep_id, train)
