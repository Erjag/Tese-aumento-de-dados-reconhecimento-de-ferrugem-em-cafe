import torch
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn import metrics

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_loss_accuracy(train_losses, val_losses, train_accs, val_accs, model_name, fold, save_dir):
    """
    Plota gráfico de loss e accuracy por epoch para conjunto de treinamento e validação
    
    Args:
        train_losses (list): Lista com os valores de loss do conjunto de treinamento por epoch
        val_losses (list): Lista com os valores de loss do conjunto de validação por epoch
        train_accs (list): Lista com os valores de accuracy do conjunto de treinamento por epoch
        val_accs (list): Lista com os valores de accuracy do conjunto de validação por epoch
    
    Returns:
        None
    """
    # Define o número de epochs
    epochs = len(train_losses)
    
    # Define o eixo x do gráfico como o número de epochs
    x = range(1, epochs + 1)
    
    # Plota os gráficos de loss e accuracy para conjunto de treinamento e validação
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, train_losses, c='magenta' ,ls='--', label='Train loss', fillstyle='none')
    plt.plot(x, val_losses, c='green' ,ls='--', label='Val. loss', fillstyle='none')
    plt.title('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, [acc.cpu() for acc in train_accs], c='magenta' ,ls='-', label='Train acuracy', fillstyle='none')
    plt.plot(x, [acc.cpu() for acc in val_accs], c='green' ,ls='-', label='Val. accuracy', fillstyle='none')
    plt.title('Accuracy')
    plt.legend()
    
    plt.savefig(f'{save_dir}/{model_name}_fold_{fold}_loss_acc.pdf')
    #plt.show()

def generate_classification_report(model, dataloader,class_names, device='gpu'):
    # Define o modelo para o dispositivo correto (CPU ou GPU)
    model = model.to(device)
    model.eval()
    
    # Inicializa as variáveis de predições e rótulos verdadeiros
    all_preds = torch.tensor([], dtype=torch.long, device=device)
    all_labels = torch.tensor([], dtype=torch.long, device=device)

    # Realiza a predição para cada lote de dados no dataloader
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        # Adiciona as predições e rótulos verdadeiros às variáveis criadas anteriormente
        all_preds = torch.cat((all_preds, preds), dim=0)
        all_labels = torch.cat((all_labels, labels), dim=0)
    
    # Gera o classification report com base nas predições e rótulos verdadeiros
    report = metrics.classification_report(all_labels.cpu().numpy(), all_preds.cpu().numpy(),target_names=class_names,
                                        digits=5, zero_division=0)
    
    return report
