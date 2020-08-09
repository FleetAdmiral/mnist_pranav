import torch
import torch.nn as nn
import io
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path
path = Path('app/pretrained.pt')

#Define the shape/architecture of the model since pre-trained model can be further worked upon, or an test image can be used to find label
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #CNN
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        #Max pool
        self.pool = nn.MaxPool2d(2, 2)
        #Neural Network
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 10)
        #Dropout Layer, with p=0.5
        self.dropout = nn.Dropout(p=.5)

    def forward(self, ans): #Once we have our test data, run it through the forward function - it takes the input through all the layers of the network
        ans = self.pool(F.relu(self.conv1(ans)))
        ans = self.pool(F.relu(self.conv2(ans)))
        ans = ans.view(-1, 64 * 7 * 7)
        ans = self.dropout(F.relu(self.fc1(ans))) #Relu helps normalise by making all negative equate to 0
        ans = self.dropout(F.relu(self.fc2(ans)))
        ans = self.dropout(F.relu(self.fc3(ans)))
        ans = F.log_softmax(self.fc4(ans), dim=1)
        return ans

#By defining a Net instance, we are now referring to a single Neural Network Net - one set of all layers layed one after the otjer
model_load = Net()
model_load.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
model_load.eval()

#How to predict - run image_bytes through the model once
def predict(image_bytes):
    transform = transforms.Compose([transforms.Resize((28,28)),
                                    transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    tensor = transform(image).unsqueeze(0)
    output = model_load.forward(tensor) #actual prediction!
    _, pred = torch.max(output, 1)
    return pred.item()
