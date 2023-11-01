from utils_libs import *
import torchvision.models as models

class client_model(nn.Module):
    def __init__(self, name, args=True):
        super(client_model, self).__init__()
        self.name = name
        if self.name == 'Linear':
            [self.n_dim, self.n_out] = args
            self.fc = nn.Linear(self.n_dim, self.n_out)
          
        if self.name == 'mnist':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 200)
            self.fc2 = nn.Linear(200, 200)
            self.fc3 = nn.Linear(200, self.n_cls)
            
        if self.name == 'emnist':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 100)
            self.fc2 = nn.Linear(100, 100)
            self.fc3 = nn.Linear(100, self.n_cls)
        
        if self.name == 'cifar10' or self.name == 'cifar10c':
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64*5*5, 384) 
            self.fc2 = nn.Linear(384, 192) 
            self.fc3 = nn.Linear(192, self.n_cls)
            
        if self.name == 'cifar100' or self.name == 'cifar100c':
            self.n_cls = 100
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64*5*5, 384) 
            self.fc2 = nn.Linear(384, 192) 
            self.fc3 = nn.Linear(192, self.n_cls)


        if self.name == 'ConvNet':
            self.n_cls = 200
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=3,padding = 1)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding = 1)
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding = 1)
            
            self.gn1 = nn.GroupNorm(4,64) 
            self.gn2 = nn.GroupNorm(4,64) 
            self.gn3 = nn.GroupNorm(4,64)

            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
             

            self.fc1 = nn.Linear(4096,512)
            self.fc2 = nn.Linear(512, 384)
            self.fc3 = nn.Linear(384, self.n_cls)
            
        if self.name == 'Resnet18':
            resnet18 = models.resnet18()
            resnet18.fc = nn.Linear(512, 100)

            # Change BN to GN 
            resnet18.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)

            assert len(dict(resnet18.named_parameters()).keys()) == len(resnet18.state_dict().keys()), 'More BN layers are there...'
            self.model = resnet18


        if self.name == 'shakespeare':
            embedding_dim = 8
            hidden_size = 100
            num_LSTM = 2
            input_length = 80
            self.n_cls = 80
            
            self.embedding = nn.Embedding(input_length, embedding_dim)
            self.stacked_LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_LSTM)
            self.fc = nn.Linear(hidden_size, self.n_cls)
              
        
    def forward(self, x):
        if self.name == 'Linear':
            x = self.fc(x)
            
        if self.name == 'mnist':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
  
        if self.name == 'emnist':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        
        if self.name == 'cifar10' or self.name=='cifar10c':
            
            x = self.conv1(x)
            x = F.relu(x)
            
            #var1 = torch.std(x,axis = [0,2,3])**2
            act1 = torch.mean(x**2,axis = [1,2,3])
            
            x = self.pool(x)
            #x = F.relu(self.conv2(x))
            x = self.conv2(x)
            x = F.relu(x)
            
            #var2 = torch.std(x,axis = [0,2,3])**2
            act2 = torch.mean(x**2,axis = [1,2,3])

            x = self.pool(x)
            
            x = x.view(-1, 64*5*5)
            #x = F.relu(self.fc1(x))
            x = self.fc1(x)
            
            x = F.relu(x)
            #var3 = torch.std(x,axis = [0])**2
            act3 = torch.mean(x**2,axis = [1])


            #x = F.relu(self.fc2(x))
            x = self.fc2(x)
            x = F.relu(x)

            #var4 = torch.std(x,axis = [0])**2
            act4 = torch.mean(x**2,axis = [1])


            x = self.fc3(x)
            
            #var5 = torch.std(x,axis = [0])**2
            #act5 = torch.mean(x**2,axis = [1])
            #var_list = [act1,act2,act3,act4,act5]
            var_list = [act1,act2,act3,act4]
        
        
        if self.name == 'cifar100' or self.name == 'cifar100c':

            #x = F.relu(self.conv1(x))
            x = self.conv1(x)
            x = F.relu(x)
            
            #var1 = torch.std(x,axis = [0,2,3])**2
            act1 = torch.mean(x**2,axis = [1,2,3])
            
            x = self.pool(x)
            #x = F.relu(self.conv2(x))
            x = self.conv2(x)
            x = F.relu(x)
            
            #var2 = torch.std(x,axis = [0,2,3])**2
            act2 = torch.mean(x**2,axis = [1,2,3])

            x = self.pool(x)
            
            x = x.view(-1, 64*5*5)
            #x = F.relu(self.fc1(x))
            x = self.fc1(x)
            
            x = F.relu(x)
            #var3 = torch.std(x,axis = [0])**2
            act3 = torch.mean(x**2,axis = [1])


            #x = F.relu(self.fc2(x))
            x = self.fc2(x)
            x = F.relu(x)

            #var4 = torch.std(x,axis = [0])**2
            act4 = torch.mean(x**2,axis = [1])


            x = self.fc3(x)
            
            #var5 = torch.std(x,axis = [0])**2
            #act5 = torch.mean(x**2,axis = [1])
            var_list = [act1,act2,act3,act4]
            #var_list = [act1,act2,act3,act4,x]

        if self.name == 'ConvNet':

            x = self.conv1(x)
            x = self.gn1(x)
            x = F.relu(x)
            
            #var1 = torch.std(x,axis = [0,2,3])**2
            var1 = torch.mean(x**2,axis = [1,2,3])

            x = self.pool(x)
            #x = F.relu(self.conv2(x))
            x = self.conv2(x)
            x = self.gn2(x)
            x = F.relu(x)
            
            #var2 = torch.std(x,axis = [0,2,3])**2
            var2 = torch.mean(x**2,axis = [1,2,3])

            x = self.pool(x)

            x = self.conv3(x)
            x = self.gn3(x)
            x = F.relu(x)
            
            x = self.pool(x)
            
            var3 = torch.mean(x**2,axis = [1,2,3])**2

            #x = x.view(-1, 64*5*5)
            x = x.view((x.shape[0], -1))
            #x = F.relu(self.fc1(x))
            x = self.fc1(x)
            
            x = F.relu(x)
            #var4 = torch.std(x,axis = [0])**2
            var3 = torch.mean(x**2,axis = [1])


            #x = F.relu(self.fc2(x))
            x = self.fc2(x)
            x = F.relu(x)

            var5 = torch.std(x,axis = [0])**2
            var4 = torch.mean(x**2,axis = [1])

            x = self.fc3(x)
            
            #var6 = torch.std(x,axis = [0])**2
            
            #var_list = [var1,var2,var3,var4,var5,var6]
            var_list = [var1,var2,var3,var4,var5]


        if self.name == 'Resnet18':
            x = self.model(x)
            var_list = None

        if self.name == 'shakespeare':
            x = self.embedding(x)
            x = x.permute(1, 0, 2) # lstm accepts in this style
            output, (h_, c_) = self.stacked_LSTM(x)
            # Choose last hidden layer
            last_hidden = output[-1,:,:]
            x = self.fc(last_hidden)

        return x,var_list,None
    
