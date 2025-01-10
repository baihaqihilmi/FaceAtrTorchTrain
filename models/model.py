from torchvision import models
from torchvision.models import resnet
from torch import nn   
class Backbone(nn.Module):
    def __init__(self,model_name,pretrained ) :
        ''''
        
        Fine tunning model 
        '''
        # get Weight 
        if pretrained == True :
            self.weight = self._get_weights(model_name)
        elif type(pretrained) == str:
            self.weight = pretrained
        self.model = self._get_model(model_name , self.weight)
        self.model = self.model.features()
        ## freezing The Model 
        for i in self.model.parameters():
            i.requires_grad = False


    def _get_model(self, model_name ,  weight = None ,):
        '''
        weights : a pretrained model 
        
        
        '''

        try:
            return getattr(models, model_name)(weight=weight)

        except AttributeError:
            raise ValueError(f"Unknown model name: {model_name}")
    def _get_weights(self , model_name ):
        try:
            w_ = model_name + "_weights"
            # Lower case all
            # weight = [x.lower() for x in weights]
            ## FIlter 
            weight = [x for x in weight if x.lower() == w_][0]
            return weight   
        except AttributeError:
            raise ValueError(f"Cannot be found: {model_name}")
    def forward(self , x):
        x = self.model(x)
        return x
class Head(nn.Module):
    def __init__(self, config ,  *args, **kwargs):
        self.feature_embeddings = 3012

        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(self.features, self.feature_embeddings)
        self.relu = nn.ReLU()
        ## Regression Problem
        self.age = nn.Linear(self.feature_embedding, 1)

        self.gender = nn.Linear(self.feature_embedding, 1)
        self.sigmoid = nn.Sigmoid()

        ## Classification Problem
        self.race = nn.Linear(self.feature_embedding, )
        self.softmax = nn.Softmax()


class Model(nn.Module):
    def __init__(self, model_name ,config):
        super(Model, self).__init__()
        self.config = config
        self.model_name = model_name
        self.features = Backbone(model_name)
        self.head = Head()

    def forward(self, x):
        x = self.features(x)
        x  = self.head.fc1(x)
        x = self.head.relu(x)

        x1 = self.head.fc1(x)

        x2 = self.head.relu(x)
        x2 = self.head.sigmoid(x)
        
        x3 = self.head.race(x)
        x3 = self.head.softmax(x)

        outputs = {  
            "Age" : x1 ,
            "Gender" :  x2,
            "Race" : x3
        }
        
        return outputs