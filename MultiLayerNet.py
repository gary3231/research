from dem_hyperelasticity.config import *


class MultiLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MultiLayerNet, self).__init__()

        # self.W = torch.nn.Parameter(torch.zeros(D_out), requires_grad=True)

        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear6 = torch.nn.Linear(H, D_out)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, H)
        # self.linear6 = torch.nn.Linear(H, H)
        # self.linear7 = torch.nn.Linear(H, D_out)

        torch.nn.init.constant_(self.linear1.bias, 0.)
        torch.nn.init.constant_(self.linear2.bias, 0.)
        torch.nn.init.constant_(self.linear3.bias, 0.)
        torch.nn.init.constant_(self.linear4.bias, 0.)
        torch.nn.init.constant_(self.linear5.bias, 0.)
        torch.nn.init.constant_(self.linear6.bias, 0.)
        # torch.nn.init.constant_(self.linear7.bias, 0.)

        # torch.nn.init.xavier_uniform_(self.linear1.weight)
        # torch.nn.init.xavier_uniform_(self.linear2.weight)
        # torch.nn.init.xavier_uniform_(self.linear3.weight)
        # torch.nn.init.xavier_uniform_(self.linear4.weight)
        # torch.nn.init.xavier_uniform_(self.linear5.weight)
        # torch.nn.init.xavier_uniform_(self.linear6.weight)

        torch.nn.init.normal_(self.linear1.weight, mean=0, std=0.1)
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=0.1)
        torch.nn.init.normal_(self.linear3.weight, mean=0, std=0.1)
        torch.nn.init.normal_(self.linear4.weight, mean=0, std=0.1)
        torch.nn.init.normal_(self.linear5.weight, mean=0, std=0.1)
        torch.nn.init.normal_(self.linear6.weight, mean=0, std=0.1)
        # torch.nn.init.normal_(self.linear7.weight, mean=0, std=0.1)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        residual = x

        y1 = torch.tanh(self.linear1(x))
        y2 = torch.tanh(self.linear2(y1))
        y3 = torch.tanh(self.linear3(y2))
        y4 = torch.tanh(self.linear4(y3))
        y5 = torch.tanh(self.linear5(y4))
        # y6 = torch.tanh(self.linear6(y5))
        # y7 = torch.tanh(self.linear6(y))
        y = (self.linear6(y5))
        # y+=residual#*self.W
        return y
