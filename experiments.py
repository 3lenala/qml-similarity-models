from train import *
from dataclasses import replace

BATCH_SIZES = [2**i for i in range(int(np.log2(dataset.len_train))+1)]
BATCH_SIZES.append(dataset.len_train) # to consider batch gradient descent too
   
LEARNING_RATES = [1,5e-1,1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4]

TOLERANCES = [1e-1, 1e-2, 1e-3, 1e-4,  1e-5, 1e-6]

PATIENCES = [1,5,10,15,20,25,30]

LAYERS=5

class HyperparametersSweep:
    def __init__(self, hyperparameter_values: list[float|int]):
        self.losses = {value: [] for value in hyperparameter_values}
        self.avg_losses = {value: [] for value in hyperparameter_values}
        self.std_losses = {value: [] for value in hyperparameter_values}
        self.time = {value: [] for value in hyperparameter_values}
        self.avg_time = {value: [] for value in hyperparameter_values}
        self.std_time = {value: [] for value in hyperparameter_values}
        self.avg_epochs = {value: [] for value in hyperparameter_values}
    
    def append_result(self, result: TrainedModel, hyperparameter_value: float | int):
        self.avg_time[hyperparameter_value].append(result.avg_time)
        self.time[hyperparameter_value].append(result.best.time)
        self.std_time[hyperparameter_value].append(result.std_time)
        self.losses[hyperparameter_value].append(result.best.best_val_loss)
        self.avg_losses[hyperparameter_value].append(result.avg_val_over_seeds)
        self.std_losses[hyperparameter_value].append(result.std_val_over_seeds)
        self.avg_epochs[hyperparameter_value].append(result.avg_epochs)
    

def run_layers(models: SQNNModel | QNNModel, dataset: PairsSet | SplittedDb):
    """Trains and evaluates each model in models for a number layers in LAYERS_LIST of data reuploading layers"""
    for i in models:
        print(f'model {i}')
        model = models[i]
        for layers in LAYERS_LIST:
            result = train_model(evaluate=True, model=model, layers=layers, dataset=dataset, seeds=SEEDS_EVALUATE)
            (model.result)[layers] = result
            print(f'L={layers}:  val loss {result.best.best_val_loss:.4f} |  train loss {result.best.final_train_loss:.4f} | '
                  f'avg loss {np.average(result.val_over_seeds):.4f}  +- {np.std(result.val_over_seeds):.4f} '
                  f' | accuracy = {result.best.accuracy:.3f} | auc {result.best.auc:.3f} | epochs {result.best.epochs}' )
    return models

def hyperparameter_sweep(models: SQNNModel | QNNModel, variable: str, values: list[int|float], dataset: PairsSet | SplittedDb, layers: int = LAYERS):
    sweep = HyperparametersSweep(values)
    for i in models:
        print(f'model {i}')
        model = models[i]
        for value in values:
            model.hyperparameters = replace(model.hyperparameters, **{variable: value})
            training_result = train_model(model=model, layers=layers, dataset=dataset, seeds=SEEDS_HYPER)
            sweep.append(training_result, value)
            print(f'{variable} = {value}:  loss {training_result.best.best_val_loss:.4f} | time {training_result.best.time:.3f}'
                  f' | avg loss {training_result.avg_val_over_seeds:4f}+-{training_result.std_val_over_seeds:4f}  | avg time {training_result.avg_time:.3f}'
                  f'+-{training_result.std_time:.3f} | avg epochs {training_result.avg_epochs:2f}+-{training_result.std_epochs}')
    return sweep

# -------------------------------------------------------------------------------------------------------
#-------------------------------------------   PLOTS   --------------------------------------------------
# -------------------------------------------------------------------------------------------------------

def hyperparameters_sweep_plot(architecture, sweep, values, xlabel, base, exp1):
    scale = base is not None
    model_labels = list(exp1)
    num_models = len(model_labels)

    plt.rcParams.update({'font.size':18})

    viridis = plt.get_cmap('viridis')
    colors = [viridis(i / max(1, num_models - 1)) for i in range(num_models+1)]

    arch_suffix = "similarity" if architecture is Architecture.SIMILARITY_FUNCTION else "overlap"

    fig3, ax = plt.subplots(figsize=(7, 5))
    for i in range(num_models):
        avg = np.array([sweep.avg_losses[item][i] for item in values])
        std = np.array([sweep.std_losses[item][i] for item in values])
        ax.plot(values, avg, marker='s', alpha=0.7,
                     color=colors[i], label=f'{model_labels[i]}', linewidth=4)
        ax.fill_between(values, avg - std, avg + std, color=colors[i], alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Average validation loss')
    if scale:
        ax.set_xscale('log', base=base)
    ax.grid(True, linestyle=':')
    plt.tight_layout()
    ax.legend(fontsize=13)
    plt.savefig(f'{xlabel}_avg_losses_{arch_suffix}.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig3)

    fig4, ax = plt.subplots(figsize=(7, 5))
    for i in range(num_models):
        avg_t = np.array([sweep.avg_time[item][i] for item in values])
        std_t = np.array([sweep.std_time[item][i] for item in values])
        ax.plot(values, avg_t,  marker='s', alpha=0.7,
                    color=colors[i], label=f'{model_labels[i]}')
        ax.fill_between(values, avg_t - std_t, avg_t + std_t, color=colors[i], alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Average training time (s)')
    if scale:
        ax.set_xscale('log', base=base)
    ax.legend(fontsize=13)
    ax.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(f'{xlabel}_avg_time_{arch_suffix}.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig4)
    
def loss_v_epochs(training_result, layers: int, model_name: str):
    plt.rcParams.update({'font.size': 28})
    
    viridis = cm.get_cmap('viridis')
    color_train = viridis(0.2) 
    color_val = viridis(0.8)
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    epochs = list(range(1, training_result.epochs + 1))
    
    ax1.plot(epochs, training_result.val_losses,
             color=color_val, marker='o', linestyle='-', linewidth=2,
             label='Validation losses')
    ax1.plot(epochs, training_result.train_losses,
             color=color_train, marker='s', linestyle='--', linewidth=2,
             label='Train losses')
    
    ax1.margins(x=0.02, y=0.05)
    
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    

    ax1.set_xlabel('Epochs',fontsize=28)
    ax1.set_ylabel('Losses',fontsize=28)
    ax1.legend(loc='best', fontsize=28)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}-{layers}layers.png', bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)
    
