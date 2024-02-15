import wandb

sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "eval/success_rate"},
    "parameters": {
        "dist_coeff": {"min": 0.01, "max": 1.0, 'distribution': 'log_uniform_values'},
        "grasp_reward": {"value": 1.0},
        "collision_penalty": {"min": 0.01, "max": 10.0, 'distribution': 'log_uniform_values'},
        "eef_position_penalty_coef": {"min": 0.01, "max": 1.0, 'distribution': 'log_uniform_values'},
        "eef_orientation_penalty_coef_relative": {"min": 0.1, "max": 10.0, 'distribution': 'log_uniform_values'},
        "regularization_coef": {"min": 0.01,  "max": 1.0, 'distribution': 'log_uniform_values'},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, entity="behavior-rl", project="sb3")
print(sweep_id)
