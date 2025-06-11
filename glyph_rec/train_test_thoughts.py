
from typing import Tuple
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
import os
from networks import ResnetClassification_General
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


from dataset import CustomImageDataset

    
    


def train_network(train_dir, num_category, num_values_all_cate, n_epochs, time, device, image_size: Tuple[int, int, int] = (3, 100, 100)):
    """
    This trains the network for a set number of epochs.
    num_category: how many parameters we have
    num_values_all_cate: list: how many values does each parameter has
    e.g., [4,2,2,4,5]
    """
    

    train_annotations_file  = os.path.join(train_dir, "train_%s.csv"%(time)) 
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    training_data = CustomImageDataset(train_annotations_file, train_dir, transform=preprocess)
    train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)

    res_model  = torchvision.models.resnet18(pretrained=True)

    model = ResnetClassification_General(res_model, num_category, num_values_all_cate)
    model.to(device)
    model.train()

   
    for name, param in model.named_parameters():
        if 'fc' in name or 'layer4' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            print(name)

  
    criterion =nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    writer = SummaryWriter()
    for epoch in range(n_epochs):

        for i, (inputs, targets, _) in enumerate(train_dataloader):
            # Zero the gradients
            optimizer.zero_grad()        
            # Forward pass
            output_predictions = model(inputs.float().to(device))

            losses = 0.0
            preds = []
            accus = []
            for para_index, para_values in enumerate(output_predictions):          
                loss = criterion(para_values, targets[:,para_index].to(device))
                losses = losses + loss
                pred = para_values.max(1).indices
                accu = torch.sum(targets[:,para_index].cuda() == pred)/pred.shape[0]
                preds.append(pred)
                accus.append(accu)
                if (i) % 10 == 0:
                    print (f'Epoch [{epoch + 1}/{n_epochs}], Step [{i}/{len(train_dataloader)}], Total Loss: {losses.item():.4f}, loss{para_index} : {loss.item():.4f}')
               
               
            # Backward pass and optimization
            losses.backward()
            optimizer.step()
            
            writer.add_scalar('Train Loss', losses.item(), i)

                
    writer.close()

    return model


def save_model(model, filename='3_100_100.pth'):
    """
    After training the model, save it so we can use it later.
    """
    torch.save(model.state_dict(), filename)


def load_model(filename, num_category, num_values_all_cate):
    """
    Load the model from the saved state dictionary.
    """
    res_model  = torchvision.models.resnet18(pretrained=True)

    model = ResnetClassification_General(res_model, num_category, num_values_all_cate)
    model.load_state_dict(torch.load(filename))
    return model



def evaluate_network(test_dir, num_category, classification_results_dir, time, model, device, image_size: Tuple[int, int, int] = (3, 100, 100)):
    """
    This evaluates the network on the test data.
    """


    test_annotations_file  = os.path.join(test_dir, "test.csv") 
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    testing_data = CustomImageDataset(test_annotations_file, test_dir, transform=preprocess)
    batch_size=32
    test_dataloader = DataLoader(testing_data, batch_size, shuffle=False)
    #only the same batch size as training works well

    criterion = nn.CrossEntropyLoss()

    model.eval()
    #https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/3
    #solution: https://discuss.pytorch.org/t/how-to-modify-a-pretrained-model/60509/10 (fix the batchnorm)
    #replace_bn(model, 'model')
    
    model.to(device)
            


    # Evaluate the model on the test data
    print("---test---")
    with torch.no_grad():
        accu_lists = [[] for _ in range(num_category)]
        loss_lists = [[] for _ in range(num_category)]
        pro_lists = [[] for _ in range(num_category)]
      
        for i, (inputs, targets, img_path) in enumerate(test_dataloader):                    
            # Forward pass
            values = model(inputs.float().to(device))
            
            losses = [criterion(values[j], targets[:, j].to(device)) for j in range(num_category)]
            for j in range(num_category):
                loss_lists[j].append(losses[j])

            preds = [values[j].max(1).indices for j in range(num_category)]
            pred_alls = [torch.topk(values[j], values[j].shape[1])[1] for j in range(num_category)]
            
            for j in range(num_category):
                accu = torch.sum(targets[:, j].cuda() == preds[j]) / preds[j].shape[0]
                accu_lists[j].append(accu)
                pro = torch.softmax(values[j], axis=-1)
                pro_lists[j].append(pro)

            print(img_path[0])
            print(*preds)

            for k in range(num_category):
                for m in range(pred_alls[k].shape[-1]):
                    pro_mean = torch.mean(pro_lists[k][i][torch.arange(preds[k].shape[0]), pred_alls[k][:, m]])
                    print(f'dim{k}, {m} th confidence {pro_mean.detach().item():.6f}')

        
    for j in range(num_category):
        np.save(f"{classification_results_dir}\\{time}_para{j}", torch.vstack(pro_lists[j]).detach().cpu().numpy())


      



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Train the model
    image_size: Tuple[int, int, int] = (3, 130, 130)
    design_dir = "."
    dataset_dir = ".\\thoughts_rec_dataset"
    realdata_dir  = "..\glyph_detection\\thoughts_detection_results"

    model_save_path = os.path.join(design_dir, "save_model") 
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    classification_results_dir = os.path.join(design_dir, "thoughts_rec_results")
    if not os.path.exists(classification_results_dir):
        os.mkdir(classification_results_dir)

    num_category=3
    num_values_all_cate=[6,2,2]
    nepoch=5

    for time in range(10): #train how many time
        
        model = train_network(dataset_dir, num_category, num_values_all_cate, nepoch, time, device, image_size=image_size)

        # Save the model
       
       
        filename = f'{image_size[0]}_{image_size[1]}_{image_size[2]}_multiple.pth'
        filename = os.path.join(model_save_path,filename)
        save_model(model, filename=filename)

        # Load the model
        model = load_model(filename, num_category, num_values_all_cate).to(device)

        # Evaluate the model
        evaluate_network(realdata_dir, num_category, classification_results_dir, time, model, device, image_size=image_size)
        