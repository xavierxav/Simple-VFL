import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch
import logging
import os
from .mlp import MLP4_Credit
from torchvision.models import resnet18

# Use the same logger as in the main file
logger = logging.getLogger(__name__)


class CompleteModel(nn.Module):
    """
    This class is a general class for a SplitNN models. It can also be adapted for aggregation models.
    
    Typically, the models_list is a list of models, where last model is the global model and the others are local models. 
    In this case, the forward method concatenates the outputs of the local models and passes them to the global model. 
    By setting the local_models to nn.Identity() and global_model to the actual model, it becomes a standard model.

    It creates an optimizer and a scheduler based on the arguments.
    """
    def __init__(self, models, args):
        super(CompleteModel, self).__init__()
        self.models = nn.ModuleList(models)
        
        weight_decay = args.get('weight_decay') or 0
        
        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=weight_decay, lr=args.lr)
        
        scheduler_config = args.get('scheduler')
        if scheduler_config is not None:
            if scheduler_config.type == 'ReduceLROnPlateau':
                self.scheduler = ReduceLROnPlateau(self.optimizer, patience=scheduler_config.patience, factor=scheduler_config.factor)
            elif scheduler_config.type == 'StepLR':
                self.scheduler = StepLR(self.optimizer, scheduler_config.step_size, scheduler_config.factor)
            else:
                raise ValueError(f'Unknown scheduler type: {scheduler_config.type}')

        continue_training = args.get('continue_training')
        if continue_training:
            self.load_model(os.path.normpath(continue_training))

    def forward(self, features):
        # Here the last model is the global model, the others are local models (but can be adapted)
        bottom_outputs = [self.models[i](features[i]) for i in range(len(self.models) - 1)]
        return bottom_outputs, self.models[-1](torch.cat(bottom_outputs, dim=1))

    def scheduler_step(self, metrics=None):
        if hasattr(self, 'scheduler'):
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(metrics)
            else:
                self.scheduler.step()
            logger.debug(f'lr={self.scheduler.get_last_lr()[0]}')

    def save_model(self, path):
        state = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None
        }
        torch.save(state, path)

    def load_model(self, path):
        state = torch.load(path, weights_only=True)
        self.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        if state['scheduler_state_dict'] is not None and hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(state['scheduler_state_dict'])

class SatelliteFeaturesCentralized(CompleteModel):
    """
    This model trains in a centralized way on features extracted from satellite sentinel images.
    Thus local models are identinty and the global model is a simple linear model.
    """
    def __init__(self, args):
        local_models = [nn.Identity() for _ in range(args.k)] #local models are identity because we are doing centralized training
        
        # Create the global model
        global_model = nn.Linear(1536*args.k, args.dataset.num_classes)
        
        super().__init__(
            models=local_models + [global_model],
            args=args
        )

class SatelliteFeaturesCentralized2layers(CompleteModel):
    """
    Same model as SatelliteFeaturesCentralized but with 2 layers in the global model.
    """
    def __init__(self, args):
        self.hidden_dim = args.hidden_dim
        # Create identity models for local models
        local_models = [nn.Identity() for _ in range(args.k)]
        
        # Create the global model
        global_model = nn.Sequential(
            nn.Linear(1536*args.k, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, args.dataset.num_classes)
        )
        
        # Call the parent class constructor
        super().__init__(
            models=local_models + [global_model],
            args=args
        )

class SpecialModel(CompleteModel):
    """
    This model first applies the same layer to all the images (with shared weights) and then apply a classical layer to the concatenated outputs.
    This was an attempt to reduce overfitting, by ussing the fact that imges are similar, but hadn't better results, than SatelliteFeaturesCentralized2layers
    """
    def __init__(self, args):
        self.k = args.k
        self.hidden_dim = args.hidden_dim
        shared_weights_layer = nn.Sequential(
            nn.Linear(1536, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU())
        
        # Create the global model
        non_shared_weights_layer = nn.Linear(self.hidden_dim*args.k, args.dataset.num_classes)
        
        super().__init__(
            models=[shared_weights_layer, non_shared_weights_layer],
            args=args
        )

    def forward(self, features):
        bottom_outputs = [self.models[0](features[i]) for i in range(self.k)] # we apply the same model to both data
        return bottom_outputs, self.models[-1](torch.cat(bottom_outputs, dim=1))

class CreditModel(CompleteModel):
    """
    Simple SplitNN model for the credit dataset, with 2 MLPs followed by a linear global model
    """
    def __init__(self, args):
        local_models = [MLP4_Credit(11, args.hidden_dim), MLP4_Credit(12, args.hidden_dim)]
        
        # Create the global model
        global_model = nn.Linear(args.hidden_dim*2, args.dataset.num_classes)

        super().__init__(
            models=local_models + [global_model],
            args=args
        )

    def forward(self, features):

        bottom_outputs = [self.models[0](features[0]) , self.models[1](features[1])] # forward method is adapted for the different shape of the input compared with satellite images
        
        return bottom_outputs, self.models[-1](torch.cat(bottom_outputs, dim=1))

class AggregationModel(CompleteModel):
    """
    Aggregation model. We don't have a global model, we average the outputs of the local models instead.
    Averaging (instead of summing) keeps the logit scale independent of the number of parties k.
    """
    def __init__(self, args):
        # Create a list of local models
        local_model = [nn.Sequential(
            nn.Linear(1536, args.hidden_dim),
            nn.BatchNorm1d(args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.dataset.num_classes)
        ) for _ in range(args.k)]
        
        super().__init__(
            models=local_model,
            args=args
        )
        self.k = args.k

    def forward(self, x):
        bottom_outputs = [self.models[i](x[i]) for i in range(self.k)]
        
        # Aggregate outputs by averaging over parties
        final_output = torch.stack(bottom_outputs).mean(dim=0)
        
        return bottom_outputs, final_output
    
class FederatedAggregationModel(CompleteModel):
    """
    In this special aggregation model we have only one model which is applied to all the images, then it averages the outputs.
    It is similar to classical federated learning but it uses all data for inference by aggregating the outputs.
    Averaging (instead of summing) keeps the logit scale independent of the number of parties k.
    """
    def __init__(self, args):
        # Create a single local model with 2 layers
        local_model = nn.Sequential(
            nn.Linear(1536, args.hidden_dim),
            nn.BatchNorm1d(args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.dataset.num_classes)
        )
        
        super().__init__(
            models=[local_model],
            args=args
        )
        self.k = args.k

    def forward(self, x):
        bottom_outputs = [self.models[0](x[i]) for i in range(self.k)]
        
        # Aggregate outputs by averaging over parties
        final_output = torch.stack(bottom_outputs).mean(dim=0)
        
        return bottom_outputs, final_output

class SPOTandSentinelAggregationModel(CompleteModel):
    """
    This model is used to train a model on SPOT and Sentinel images.
    It applies the same model on all sentinel images and another one on SPOT image.
    It then sums the outputs of the two models.
    """
    def __init__(self, args):
        self.hidden_dim = args.hidden_dim
        self.num_classes = args.dataset.num_classes
        self.sentinel_weight = args.sentinel_weight
        # Create a single local model with 2 layers
        local_model_sentinel = nn.Sequential(
            nn.Linear(1536, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )
        local_model_spot = nn.Sequential(
            nn.Linear(1536, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )
        
        super().__init__(
            models=[local_model_sentinel, local_model_spot],
            args=args
        )
        self.k = args.k

    def forward(self, x):
        bottom_outputs = []
        bottom_outputs.append(self.models[0](x[0]))
        for i in range(1, self.k):
            bottom_outputs.append(self.models[1](x[i]) / (self.k - 1) * self.sentinel_weight)
        final_output = torch.sum(torch.stack(bottom_outputs, dim=0), dim=0)
        return bottom_outputs, final_output

class VertibenchReproduction(CompleteModel):
    """
    This model is used to train a model on Vertibench dataset. It is a simple model with 2 layers.
    """
    def __init__(self, args, kernel_size=9, channels=13):
        # Create a single local model with 2 layers
        self.k = args.k
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.dataset.num_classes
        local_models = []
        self.cut_dim = 0
        for _ in range(self.k):
            resnet = resnet18(weights=None)
            self.cut_dim += resnet.fc.in_features
            resnet.fc = nn.Identity()
            resnet.conv1 = nn.Conv2d(channels, 64, kernel_size, stride=2, padding=3, bias=False)
            local_models.append(resnet)

        super().__init__(
            models=local_models + [self.MLP()],
            args=args
        )
    
    def MLP(self):
        return nn.Sequential(
            nn.Linear(self.cut_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )


    def forward(self, Xs):
        local_outputs = [resnet(Xi.float()) for resnet, Xi in zip(self.models[:-1], Xs)]
        agg_input = torch.cat(local_outputs, dim=1)
        return local_outputs, self.models[-1](agg_input)
    
