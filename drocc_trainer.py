import os
import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import pandas as pd

#trainer class for DROCC
class DROCCTrainer:
    """
    Trainer class that implements the DROCC algorithm proposed in
    https://arxiv.org/abs/2002.12718
    """

    def __init__(self, model, optimizer, lamda, radius, gamma, device):
        """Initialize the DROCC Trainer class
        Parameters
        ----------
        model: Torch neural network object
        optimizer: Total number of epochs for training.
        lamda: Weight given to the adversarial loss
        radius: Radius of hypersphere to sample points from.
        gamma: Parameter to vary projection.
        device: torch.device object for device to use.
        """
        self.model = model
        self.optimizer = optimizer
        self.lamda = lamda
        self.radius = radius
        self.gamma = gamma
        self.device = device

    def train(self, normal_class, seed, train_loader, val_loader, learning_rate, lr_scheduler, total_epochs,
                only_ce_epochs=50, ascent_step_size=0.001, ascent_num_steps=50,
                metric='AUC'):
        """Trains the model on the given training dataset with periodic
        evaluation on the validation dataset.
        Parameters
        ----------
        train_loader: Dataloader object for the training dataset.
        val_loader: Dataloader object for the validation dataset.
        learning_rate: Initial learning rate for training.
        total_epochs: Total number of epochs for training.
        only_ce_epochs: Number of epochs for initial pretraining.
        ascent_step_size: Step size for gradient ascent for adversarial
                          generation of negative points.
        ascent_num_steps: Number of gradient ascent steps for adversarial
                          generation of negative points.
        metric: Metric used for evaluation (AUC / F1).
        """
        best_score = -np.inf
        best_model = None
        patience = 10
        pat = 0
        self.ascent_num_steps = ascent_num_steps
        self.ascent_step_size = ascent_step_size
        for epoch in range(total_epochs):
            #Make the weights trainable
            self.model.train()
            lr_scheduler(epoch, total_epochs, only_ce_epochs, learning_rate, self.optimizer)

            #Placeholder for the respective 2 loss values
            epoch_adv_loss = torch.tensor([0]).type(torch.float32).to(self.device)  #AdvLoss
            epoch_ce_loss = 0  #Cross entropy Loss

            batch_idx = -1
            for data, target, _ in train_loader:
                batch_idx += 1
                data, target = data.to(self.device), target.to(self.device)
                # Data Processing
                data = data.to(torch.float)
                target = target.to(torch.float)
                target = torch.squeeze(target)
                print(len(targets))

                self.optimizer.zero_grad()

                # Extract the logits for cross entropy loss
                logits = self.model(data)
                logits = torch.squeeze(logits, dim = 1)
                ce_loss = F.binary_cross_entropy_with_logits(logits, target)
                # Add to the epoch variable for printing average CE Loss
                epoch_ce_loss += ce_loss

                '''
                Adversarial Loss is calculated only for the positive data points (label==1).
                '''
                if  epoch >= only_ce_epochs:
                    data = data[target == 1]
                    # AdvLoss
                    adv_loss = self.one_class_adv_loss(data)
                    epoch_adv_loss += adv_loss

                    loss = ce_loss + adv_loss * self.lamda
                else:
                    # If only CE based training has to be done
                    loss = ce_loss

                # Backprop
                loss.backward()
                self.optimizer.step()

            epoch_ce_loss = epoch_ce_loss/(batch_idx + 1)  #Average CE Loss
            epoch_adv_loss = epoch_adv_loss/(batch_idx + 1) #Average AdvLoss

            test_score, auc, df  = self.test(val_loader, metric)
            if test_score > best_score:
                pat=0
                best_score = test_score
                best_epoch = epoch
                best_model = copy.deepcopy(self.model)

                df.to_csv('data.csv')
                output_name = 'results_norm_class_' + str(normal_class) + '_seed_' + str(seed)
                pd.DataFrame([[test_score, auc]]).to_csv(output_name)
            else:
                pat+=1
            print('Epoch: {}, CE Loss: {}, AdvLoss: {}, {}: {}'.format(
                epoch, epoch_ce_loss.item(), epoch_adv_loss.item(),
                metric, test_score))

            if pat ==patience:
              break
        self.model = copy.deepcopy(best_model)
        print('\nBest test {}: {} on epoch {}'.format(
            metric, best_score, best_epoch
        ))
        return best_score, best_epoch

    def test(self, test_loader, metric):
        """Evaluate the model on the given test dataset.
        Parameters
        ----------
        test_loader: Dataloader object for the test dataset.
        metric: Metric used for evaluation (AUC / F1).
        """
        self.model.eval()
        label_score = []
        batch_idx = -1
        for data, target, _ in test_loader:
            batch_idx += 1
            data, target = data.to(self.device), target.to(self.device)
            data = data.to(torch.float)
            target = target.to(torch.float)
            target = torch.squeeze(target)

            logits = self.model(data)
            logits = torch.squeeze(logits, dim = 1)
            sigmoid_logits = torch.sigmoid(logits)
            scores = logits
            label_score += list(zip(target.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))
        # Compute test score
        labels, scores = zip(*label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        if metric == 'F1':
            # Evaluation based on https://openreview.net/forum?id=BJJLHbb0-
            thresh = np.percentile(scores, 90)
            y_pred = np.where(scores >= thresh, 1, 0)
            prec, recall, test_metric, _ = precision_recall_fscore_support(
                labels, y_pred, average="binary")



            df=pd.concat([pd.DataFrame(scores), pd.DataFrame(labels), pd.DataFrame(y_pred)], axis =1)
            df.columns = ['output', 'label', 'pred']
            auc= roc_auc_score(labels, scores)
            print('AUC is {}'.format(roc_auc_score(labels, scores)))
            print('prec is {}'.format(prec))
            print('recall is {}'.format(recall))


            df['label'] = 1 - df['label']
            df['output'] = df['output'] * -1
            thresh = np.percentile(df['output'], 10)
            prec = ( len(df.loc[(df['output'] > thresh) & (df['label'] == 1)]) ) / ( len(df.loc[(df['output'] > thresh) & (df['label'] == 1)]) + len(df.loc[(df['output'] > thresh) & (df['label'] == 0)]))
            print(prec)
            recall = prec = ( len(df.loc[(df['output'] > thresh) & (df['label'] == 1)]) ) / ( len(df.loc[(df['output'] > thresh) & (df['label'] == 1)]) + len(df.loc[(df['output'] <= thresh) & (df['label'] == 1)]))
            print(recall)
            print('correct f1 score {}' .format(2 * ((prec * recall ) / (prec + recall))))
            test_metric = 2 * ((prec * recall ) / (prec + recall))

        if metric == 'AUC':
            df=pd.DataFrame([[0]])
            test_metric = roc_auc_score(labels, scores)
        return test_metric, auc, df


    def one_class_adv_loss(self, x_train_data):
        """Computes the adversarial loss:
        1) Sample points initially at random around the positive training
            data points
        2) Gradient ascent to find the most optimal point in set N_i(r)
            classified as +ve (label=0). This is done by maximizing
            the CE loss wrt label 0
        3) Project the points between spheres of radius R and gamma * R
            (set N_i(r))
        4) Pass the calculated adversarial points through the model,
            and calculate the CE loss wrt target class 0

        Parameters
        ----------
        x_train_data: Batch of data to compute loss on.
        """
        batch_size = len(x_train_data)
        # Randomly sample points around the training data
        # We will perform SGD on these to find the adversarial points
        x_adv = torch.randn(x_train_data.shape).to(self.device).detach().requires_grad_()
        x_adv_sampled = x_adv + x_train_data

        for step in range(self.ascent_num_steps):
            with torch.enable_grad():

                new_targets = torch.zeros(batch_size, 1).to(self.device)
                new_targets = torch.squeeze(new_targets)
                new_targets = new_targets.to(torch.float)

                logits = self.model(x_adv_sampled)
                logits = torch.squeeze(logits, dim = 1)
                new_loss = F.binary_cross_entropy_with_logits(logits, new_targets)

                grad = torch.autograd.grad(new_loss, [x_adv_sampled])[0]
                grad_norm = torch.norm(grad, p=2, dim = tuple(range(1, grad.dim())))
                grad_norm = grad_norm.view(-1, *[1]*(grad.dim()-1))
                grad_normalized = grad/grad_norm
            with torch.no_grad():
                x_adv_sampled.add_(self.ascent_step_size * grad_normalized)

            if (step + 1) % 10==0:
                # Project the normal points to the set N_i(r)
                h = x_adv_sampled - x_train_data
                norm_h = torch.sqrt(torch.sum(h**2,
                                                dim=tuple(range(1, h.dim()))))
                alpha = torch.clamp(norm_h, self.radius,
                                    self.gamma * self.radius).to(self.device)
                # Make use of broadcast to project h
                proj = (alpha/norm_h).view(-1, *[1] * (h.dim()-1))
                h = proj * h
                x_adv_sampled = x_train_data + h  #These adv_points are now on the surface of hyper-sphere

        adv_pred = self.model(x_adv_sampled)
        adv_pred = torch.squeeze(adv_pred, dim=1)
        adv_loss = F.binary_cross_entropy_with_logits(adv_pred, (new_targets * 0))

        return adv_loss

    def save(self, path):
        torch.save(self.model.state_dict(),os.path.join(path, 'model.pt'))

    def load(self, path):
        self.model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))
