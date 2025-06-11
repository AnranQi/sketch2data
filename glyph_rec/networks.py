import torch.nn as nn
from torchvision.models import resnet50
  
import timm
class CNNRegression(nn.Module):
    """
    This will be the very basic CNN model we will use for the regression task.
    """
    def __init__(self, image_size=(3, 100, 100)):
        super(CNNRegression, self).__init__()
        self.image_size = image_size
        self.conv1 = nn.Conv2d(in_channels=self.image_size[0], out_channels=4, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear_line_size = int(16*(image_size[1]//4)*(image_size[2]//4))
        self.fc1 = nn.Linear(in_features=self.linear_line_size, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=4)

        
    def forward(self, x):
        """
        Passes the data through the network.
        There are commented out print statements that can be used to 
        check the size of the tensor at each layer. These are very useful when
        the image size changes and you want to check that the network layers are 
        still the correct shape.
        """
        x = self.conv1(x)
        # print('Size of tensor after each layer')
        # print(f'conv1 {x.size()}')
        x = nn.functional.relu(x)
        # print(f'relu1 {x.size()}')
        x = self.pool1(x)
        # print(f'pool1 {x.size()}')
        x = self.conv2(x)
        # print(f'conv2 {x.size()}')
        x = nn.functional.relu(x)
        # print(f'relu2 {x.size()}')
        x = self.pool2(x)
        # print(f'pool2 {x.size()}')
        x = x.view(-1, self.linear_line_size)
        # print(f'view1 {x.size()}')
        x = self.fc1(x)
        # print(f'fc1 {x.size()}')
        x = nn.functional.relu(x)
        # print(f'relu2 {x.size()}')
        x = self.fc2(x)
        # print(f'fc2 {x.size()}')
        return x

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x


class ResnetClassification_Lollipop(nn.Module): 
    def __init__(self, resnet_model, num_class0, num_class1, num_class2, num_class3):
        super(ResnetClassification_Lollipop, self).__init__()
        # num_classes= 4 # Replace num_classes with the number of classes in your data

        # Load pre-trained model from timm
        self.res_model = resnet_model
        self.num_features = self.res_model.fc.in_features
        self.res_model.fc = Identity()
        # check weights: self.res_model.conv1.weight[0,0]
        # Modify the model head for fine-tuning
      
        # self.model = torch.nn.Sequential(*(list(self.res_model.children())[:-1])) #remove the fc layer (classification layer) in the original resnet
        # check weights: self.model[0].weight[0,0] (as this is sequential, so it is not called layer ... anymore), but the weights are the same as the pretained resnet
        # Additional linear layer and dropout layer
        self.fc_para0 = nn.Sequential(
            nn.Linear(self.num_features, 256),  # Additional linear layer with 256 output features
            nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
            nn.Dropout(0.5),               # Dropout layer with 50% probability
            nn.Linear(256, 128),    # Final prediction fc layer
            nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
            nn.Dropout(0.5), 
            nn.Linear(128, num_class0)
        
        )

        self.fc_para1 = nn.Sequential(
            nn.Linear(self.num_features, 256),  # Additional linear layer with 256 output features
            nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
            nn.Dropout(0.5),               # Dropout layer with 50% probability
            nn.Linear(256, 128),    # Final prediction fc layer
            nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
            nn.Dropout(0.5), 
            nn.Linear(128, num_class1)
        )

        self.fc_para2 = nn.Sequential(
            nn.Linear(self.num_features, 256),  # Additional linear layer with 256 output features
            nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
            nn.Dropout(0.5),               # Dropout layer with 50% probability
            nn.Linear(256, 128),    # Final prediction fc layer
            nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
            nn.Dropout(0.5), 
            nn.Linear(128, num_class2)
        )

        self.fc_para3 = nn.Sequential(
            nn.Linear(self.num_features, 256),  # Additional linear layer with 256 output features
            nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
            nn.Dropout(0.5),               # Dropout layer with 50% probability
            nn.Linear(256, 128),    # Final prediction fc layer
            nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
            nn.Dropout(0.5), 
            nn.Linear(128, num_class3)
        )

        # Freeze specific layers (e.g.,the first two convolutional layers) of the pre-trained model
     


    def forward(self, x):


        x = self.res_model(x)

        value0 = self.fc_para0(x)
        value1 = self.fc_para1(x)
        value2 = self.fc_para2(x)
        value3 = self.fc_para3(x)

    

        return value0, value1, value2, value3
      
        

class ResnetClassification_Triangle(nn.Module): 
    def __init__(self, resnet_model, num_class0, num_class1, num_class2, num_class3, num_class4):
        super(ResnetClassification_Triangle, self).__init__()
        # num_classes= 4 # Replace num_classes with the number of classes in your data

        # Load pre-trained model from timm
        self.res_model = resnet_model
        self.num_features = self.res_model.fc.in_features
        
        self.res_model.fc = Identity()
        # check weights: self.res_model.conv1.weight[0,0]
        # Modify the model head for fine-tuning
      
        # self.model = torch.nn.Sequential(*(list(self.res_model.children())[:-1])) #remove the fc layer (classification layer) in the original resnet
        # check weights: self.model[0].weight[0,0] (as this is sequential, so it is not called layer ... anymore), but the weights are the same as the pretained resnet
        # Additional linear layer and dropout layer
        self.fc_para0 = nn.Sequential(
            nn.Linear(self.num_features, 256),  # Additional linear layer with 256 output features
            nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
            nn.Dropout(0.5),               # Dropout layer with 50% probability
            nn.Linear(256, 128),    # Final prediction fc layer
            nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
            nn.Dropout(0.5), 
            nn.Linear(128, num_class0)
        
        )

        self.fc_para1 = nn.Sequential(
            nn.Linear(self.num_features, 256),  # Additional linear layer with 256 output features
            nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
            nn.Dropout(0.5),               # Dropout layer with 50% probability
            nn.Linear(256, 128),    # Final prediction fc layer
            nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
            nn.Dropout(0.5), 
            nn.Linear(128, num_class1)
        )

        self.fc_para2 = nn.Sequential(
            nn.Linear(self.num_features, 256),  # Additional linear layer with 256 output features
            nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
            nn.Dropout(0.5),               # Dropout layer with 50% probability
            nn.Linear(256, 128),    # Final prediction fc layer
            nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
            nn.Dropout(0.5), 
            nn.Linear(128, num_class2)
        )

        self.fc_para3 = nn.Sequential(
            nn.Linear(self.num_features, 256),  # Additional linear layer with 256 output features
            nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
            nn.Dropout(0.5),               # Dropout layer with 50% probability
            nn.Linear(256, 128),    # Final prediction fc layer
            nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
            nn.Dropout(0.5), 
            nn.Linear(128, num_class3)
        )

        self.fc_para4 = nn.Sequential(
            nn.Linear(self.num_features, 256),  # Additional linear layer with 256 output features
            nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
            nn.Dropout(0.5),               # Dropout layer with 50% probability
            nn.Linear(256, 128),    # Final prediction fc layer
            nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
            nn.Dropout(0.5), 
            nn.Linear(128, num_class4)
        )

        # Freeze specific layers (e.g.,the first two convolutional layers) of the pre-trained model
     


    def forward(self, x):


        x = self.res_model(x)

        value0 = self.fc_para0(x)
        value1 = self.fc_para1(x)
        value2 = self.fc_para2(x)
        value3 = self.fc_para3(x)
        value4 = self.fc_para4(x)

    

        return value0, value1, value2, value3, value4



class ResnetClassification_General(nn.Module): 
    def __init__(self, resnet_model, num_category, num_values_all_cate):
        super(ResnetClassification_General, self).__init__()
        # num_classes= 4 # Replace num_classes with the number of classes in your data

        # Load pre-trained model from timm
        self.res_model = resnet_model
        self.num_features = self.res_model.fc.in_features #512
       
        self.res_model.fc = Identity()
        # check weights: self.res_model.conv1.weight[0,0]
        # Modify the model head for fine-tuning
      
        # self.model = torch.nn.Sequential(*(list(self.res_model.children())[:-1])) #remove the fc layer (classification layer) in the original resnet
        # check weights: self.model[0].weight[0,0] (as this is sequential, so it is not called layer ... anymore), but the weights are the same as the pretained resnet
        # Additional linear layer and dropout layer
        self.fc_layers = nn.ModuleList()
        for num_class in range(num_category):
            fc_layer = nn.Sequential(
                nn.Linear(self.num_features, 256),  # Additional linear layer with 256 output features
                nn.ReLU(inplace=True),         # Activation function
                nn.Dropout(0.5),               # Dropout layer with 50% probability
                nn.Linear(256, 128),           # Additional linear layer with 128 output features
                nn.ReLU(inplace=True),         # Activation function
                nn.Dropout(0.5),               # Dropout layer with 50% probability
                nn.Linear(128, num_values_all_cate[num_class])      # Final prediction fc layer
            )
            self.fc_layers.append(fc_layer)
     


    def forward(self, x):


        x = self.res_model(x)

        outputs = []
        for fc_layer in self.fc_layers:
            outputs.append(fc_layer(x))
        
        return tuple(outputs)

    

      