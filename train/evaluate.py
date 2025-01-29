import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

def evaluate_model(model, test_loader, num_samples=10):
    """
    Evalue la performance d'un modèle PyroBayesian1DCNN sur test_loader.
    Retourne l'accuracy.
    On fait num_samples forward pour estimer l'incertitude des poids.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            sample_preds = []
            for _ in range(num_samples):
                logits = model.forward(x_batch).squeeze(-1)
                prob = torch.sigmoid(logits)
                sample_preds.append(prob.unsqueeze(0))
                
            # Moyenne des modèles samplés
            mean_preds = torch.cat(sample_preds, dim=0).mean(dim=0)
            pred_classes = (mean_preds > 0.5).int() # On pourrait imaginer de régler le seuil
            
            all_preds.append(pred_classes)
            all_targets.append(y_batch.int())
    
    y_pred = torch.cat(all_preds).cpu().numpy()
    y_true = torch.cat(all_targets).cpu().numpy()
    
    acc = f1_score(y_true, y_pred)
    return acc
