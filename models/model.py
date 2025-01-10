from torchvision import models
from torchvision.models import resnet50
from torch import nn   
import torch
class Backbone(nn.Module):
    def __init__(self,model_name,pretrained = False , freeze = False) :
        super(Backbone, self).__init__()

        ''''
        
        Fine tunning model 
        '''

        self.freeze = freeze
        # get Weight 
        self.weight = None
        if pretrained == True :
            self.weight = self._get_weights(model_name)
        elif type(pretrained) == str:
            self.weight = pretrained
        self.model = self._get_model(model_name , self.weight)
        self.model = self.model
        print(self.model)
        if hasattr(self.model, 'features'):
            self.model = self.model.features
        else:
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        ## freezing The Model 
        if self.freeze :
            for i in self.model.parameters():
                i.requires_grad = False



    def _get_model(self, model_name ,  weight = None ,):
        '''
        weights : a pretrained model 
        
        
        '''

        try:
            if weight:
                return getattr(models, model_name)( weight=weight )
            else : 
                return getattr(models, model_name)()
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
class Head(nn.Module): ## New learning Device neet to go To Cuda
    def __init__(self,  num_features):
        super(Head, self).__init__()
        self.feature_embeddings = 3028
        self.fc = nn.Linear(num_features, self.feature_embeddings)  # Adjust 2048 to match output channels of the feature extractor

        ## Regression Problem
        self.age = nn.Linear(self.feature_embeddings, 1)

        self.gender = nn.Linear(self.feature_embeddings, 1)
        self.sigmoid = nn.Sigmoid()

        ## Classification Problem
        self.race = nn.Linear(self.feature_embeddings,  5)
        self.softmax = nn.Softmax()


class MyModel(nn.Module):
    def __init__(self, args ):
        super(MyModel, self).__init__()
        self.args = args
        self.model_name = args.model_name
        self.num_features = 0
        self.features = Backbone(self.model_name)
        self.get_dummy()
        self.head = Head(self.num_features)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.head.fc(x)
        x = torch.relu(x)

        x1 = self.head.age(x)

        x2 = self.head.gender(x)
        
        x3 = self.head.race(x)

        outputs = {  
            "Age" : x1 ,
            "Gender" :  x2,
            "Race" : x3
        }
        
        return outputs
    def get_dummy(self):
        dummy_input = torch.randn(1, 3 , *self.args.img_size)  # Batch size of 1, typical input shape
        with torch.no_grad():
            feature_output = self.features(dummy_input)
        self.num_features = feature_output.shape[1]  # Number of channels

if __name__ =="__main__":
    args = { 
        "model_name" : 'resnet50' ,
        'img_size' : [224,224]


    }
    model = MyModel(args)
    random_image = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 color channels, 224x224 image
    output = model(random_image)
    print(output)
