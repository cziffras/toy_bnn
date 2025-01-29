import optuna
import pyro
import torch
from train.train import train_model
from train.evaluate import evaluate_model
from model.pyrobayesian1dcnn import PyroBayesian1DCNN
from data.mushroom_data import load_mushroom_data

def objective(trial):
    """
    Objectif Optuna : on reçoit un 'trial' sur lequel on échantillonne des hyperparamètres,
    on crée le modèle, on l'entraîne, on l'évalue, et on retourne l'accuracy (ou autre).
    """
    # On suggère des hyperparamètres
    proj_channels = trial.suggest_categorical("proj_channels", [2, 4, 8, 16])
    proj_seq_len = trial.suggest_int("proj_seq_len", 4, 16)  
    conv_out_channels = trial.suggest_categorical("conv_out_channels", [16, 32, 64])
    kernel_size = trial.suggest_int("kernel_size", 3, 7)
    prior_std = trial.suggest_loguniform("prior_std", 1e-2, 1.0)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    num_epochs = trial.suggest_int("num_epochs", 5, 15)
    
    # On recharge les DataLoaders (pour s'assurer de tout réinitialiser)
    train_loader, test_loader, input_dim = load_mushroom_data(
        batch_size=64,
        test_size=0.2,
        random_state=42
    )
    
    # On instancie le modèle PyroBayesian1DCNN
    model = PyroBayesian1DCNN(
        input_dim=input_dim,
        output_dim=1,
        proj_channels=proj_channels,
        proj_seq_len=proj_seq_len,
        conv_out_channels=conv_out_channels,
        kernel_size=kernel_size,
        prior_std=prior_std
    )

    # IMPORTANT : Réinitialiser les seeds pour rendre le process reproductible
    pyro.set_rng_seed(42)
    torch.manual_seed(42)

    # On entraîne
    train_model(
        model=model,
        train_loader=train_loader,
        num_epochs=num_epochs,
        lr=lr,
        num_particles=1
    )

    # On évalue
    f1 = evaluate_model(model, test_loader, num_samples=5)  
    
    return f1


def run_hyperparam_optimization(n_trials=20):
    """
    Lance l'optimisation Optuna avec n_trials essais,
    et retourne la meilleure accuracy + best_params.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Best f1: {study.best_value}")
    print(f"Best trial params: {study.best_params}")
    return study
