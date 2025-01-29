import torch
import pyro
import pyro.infer
import pyro.optim as optim

def train_model(model, train_loader, num_epochs=10, lr=0.001, num_particles=1):
    """
    Entraîne le modèle PyroBayesian1DCNN sur un DataLoader donné.
    Retourne l'objet SVI (stochastic variational inference) entraîné.
    """

    # Problème avec le déplacement du modèle sur cuda, checker si les tenseurs lors du forward sont bien sur gpu
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
    device = torch.device("cpu")
    print(f"Utilisation de : {device}")

    model.to(device)
    for name, param in model.named_parameters():
        print(f"{name}: {param.device}")

 
    adam_params = {"lr": lr}
    optimizer = optim.Adam(adam_params)

    svi = pyro.infer.SVI(
        model=model.model,
        guide=model.guide,
        optim=optimizer,
        loss=pyro.infer.Trace_ELBO(num_particles=num_particles)
    ) 

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train() 
        for x_batch, y_batch in train_loader:
 
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
     
            loss = svi.step(x_batch, y_batch)
            epoch_loss += loss
        
        epoch_loss /= len(train_loader.dataset)
        print(f"[Epoch {epoch+1}] Loss: {epoch_loss:.4f}")
    
    return svi
