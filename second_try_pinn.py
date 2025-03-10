# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 10:18:29 2022

@author: User
"""

# Import necessary libraries
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import time
from torch import autograd
import os
import math

torch.cuda.empty_cache()
#cuda0 = torch.device('cuda:0')
start = time.time()
class Net(nn.Module):
    def __init__(self, num_layers, num_neurons):
        super(Net, self).__init__()
        
        # Define the network architecture
        self.input_layer = nn.Linear(2, num_neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(num_neurons, num_neurons) for i in range(num_layers)])
        self.output_layer = nn.Linear(num_neurons, 3)
     
    def forward(self, x):
        # Propagate input through the network
        o = self.act(self.input_layer(x))
        for i, li in enumerate(self.hidden_layers):
            o = self.act(li(o))
        out = self.output_layer(o)
        return out
    
    def act(self, x):
        # Activation function
        return torch.tanh(x)


# Define the domain of the function
x_min, x_max = 0, 1
y_min, y_max = 0, 1

# Define the grid of points at which to evaluate the function
size=50
x = torch.linspace(x_min, x_max, size).cuda()
y = torch.linspace(y_min, y_max, size).cuda()
x, y = torch.meshgrid(x, y)

# Define the points on the boundaries
x_boundary_top = torch.stack((x[0, :], y[0, :]), dim=1)
x_boundary_bottom = torch.stack((x[-1, :], y[-1, :]), dim=1)
x_boundary_left = torch.stack((x[:, 0], y[:, 0]), dim=1)
x_boundary_right = torch.stack((x[:, -1], y[:, -1]), dim=1)


# Concatenate the boundary points
boundary = torch.cat((x_boundary_top, x_boundary_bottom, x_boundary_left, x_boundary_right), dim=0)


# # Define the viscosity
mu = 1


# Flatten the inputs
x = x.flatten()
y = y.flatten()
x = Variable(x, requires_grad=True)
y = Variable(y, requires_grad=True)


# Concatenate the inputs into a single tensor
inputs = torch.stack((x, y), dim=1)
inputs = Variable(inputs , requires_grad = True)


fx= -np.pi*torch.cos(np.pi*x)*torch.sin(np.pi*y)+2*(np.pi**3) *torch.cos(2*np.pi*x)*torch.sin(2*np.pi*y)-4*(np.pi**3) * (torch.sin(np.pi*x)**2) *torch.sin(np.pi*2*y)
fy= -np.pi*torch.cos(np.pi*y)*torch.sin(np.pi*x)-2*(np.pi**3) *torch.cos(2*np.pi*y)*torch.sin(2*np.pi*x)+4*(np.pi**3) * (torch.sin(np.pi*y)**2) *torch.sin(np.pi*2*x)





f_1=fx.flatten().unsqueeze(-1)
f_2=fy.flatten().unsqueeze(-1)


# Define the network
net = Net(num_layers=5, num_neurons=45)
net=net.cuda()
# Define the optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)#1e-5

# Define the loss function
loss_fn = nn.MSELoss()
loss_fn = loss_fn.cuda()



# Train the network
for i in range(100001):
    
    optimizer.zero_grad()
    pred_input=net(inputs)
    
    p_d =autograd.grad(outputs=pred_input[:,2], inputs=inputs,grad_outputs=torch.ones_like(pred_input[:,2]), create_graph=True)
    p_dx = p_d[0][:, 0].unsqueeze(-1)
    p_dy = p_d[0][:, 1].unsqueeze(-1)
    #p_dxx = autograd.grad(outputs=p_dx, inputs=inputs,grad_outputs=torch.ones_like(p_dx), create_graph=True)[0][:, 0].unsqueeze(-1)
    #p_dyy = autograd.grad(outputs=p_dy, inputs=inputs,grad_outputs=torch.ones_like(p_dy), create_graph=True)[0][:, 1].unsqueeze(-1) 
    
    u_d =autograd.grad(outputs=pred_input[:,0], inputs=inputs,grad_outputs=torch.ones_like(pred_input[:,0]), create_graph=True)
    u_dx = u_d[0][:, 0].unsqueeze(-1)
    u_dy = u_d[0][:, 1].unsqueeze(-1)
    u_ddx = autograd.grad(outputs=u_dx, inputs=inputs,grad_outputs=torch.ones_like(u_dx), create_graph=True)
    u_dxx = u_ddx[0][:, 0].unsqueeze(-1)
    u_dxy = u_ddx[0][:, 1].unsqueeze(-1)
    u_ddy = autograd.grad(outputs=u_dy, inputs=inputs,grad_outputs=torch.ones_like(u_dy), create_graph=True)
    u_dyx = u_ddy[0][:, 0].unsqueeze(-1)
    u_dyy = u_ddy[0][:, 1].unsqueeze(-1)
    
    v_d =autograd.grad(outputs=pred_input[:,1], inputs=inputs,grad_outputs=torch.ones_like(pred_input[:,1]), create_graph=True)
    v_dx = v_d[0][:, 0].unsqueeze(-1)
    v_dy = v_d[0][:, 1].unsqueeze(-1)
    v_ddx = autograd.grad(outputs=v_dx, inputs=inputs,grad_outputs=torch.ones_like(v_dx), create_graph=True)
    v_dxx = v_ddx[0][:, 0].unsqueeze(-1)
    v_dxy = v_ddx[0][:, 1].unsqueeze(-1)
    v_ddy = autograd.grad(outputs=v_dy, inputs=inputs,grad_outputs=torch.ones_like(v_dy), create_graph=True)
    v_dyx = v_ddy[0][:, 0].unsqueeze(-1)
    v_dyy = v_ddy[0][:, 1].unsqueeze(-1)
    
    # # Compute the loss  

    l2=(-p_dx)+u_dxx+u_dyy-f_1.cuda()
    l3=(-p_dy)+v_dxx+v_dyy-f_2.cuda()
    loss2 = loss_fn(l2,torch.zeros([len(l2),1]).cuda())
    loss2 = (loss2/(len(l2))).cuda()
    loss3 = loss_fn(l3,torch.zeros([len(l3),1]).cuda())
    loss3 = (loss3/(len(l3))).cuda()

    #邊界值條件
    
    boundary_net = net(boundary)
    
    loss4 = loss_fn(boundary_net[:,0],torch.zeros([size,1]).cuda())
    loss4 = (loss4/(len(boundary_net))).cuda()
    loss5 = loss_fn(boundary_net[:,1],torch.zeros([size,1]).cuda())
    loss5 = (loss5/(len(boundary_net))).cuda()
    #laplace u =0

    loss8 = loss_fn(u_dx+v_dy,torch.zeros([len(u_dx),1]).cuda())
    loss8 = (loss8/(len(u_dx))).cuda()
    
    loss = loss2 + loss3 + loss4 + loss5  +loss8
    
    # Backpropagate the loss
    loss = loss.mean()
    loss.backward(retain_graph=True)
    
    # Update the network weights
    optimizer.step()
    path='D:/研究所/科學計算/實驗資料/point900_p'
    name_list=[]
    j=0
    # Print the loss
    if i <10001:
        
        if i % 1000 == 0:
            print(f'step: {i} loss = {loss.item()}')
            
            u_pred =net(inputs)[:,0].cpu().reshape(size,size).detach().numpy()
    
            v_pred =net(inputs)[:,1].cpu().reshape(size,size).detach().numpy()
    
            p_pred =net(inputs)[:,2].cpu().reshape(size,size).detach().numpy()
            
            plt.subplot(2,2,1)
            plt.imshow(u_pred)
            plt.colorbar()
            plt.subplot(2,2,2)
            plt.imshow(v_pred)
            plt.colorbar()
            plt.subplot(2,2,3)
            plt.imshow(p_pred)
            plt.colorbar()
            name_list.append(str(i))
            plt.savefig(os.path.join(path,name_list[j]))
            plt.show()
            j+=1
    elif i>10000 and i <100000:
        if i % 10000 == 0:
            print(f'step: {i} loss = {loss.item()}')
            u_pred =net(inputs)[:,0].cpu().reshape(size,size).detach().numpy()
    
            v_pred =net(inputs)[:,1].cpu().reshape(size,size).detach().numpy()
    
            p_pred =net(inputs)[:,2].cpu().reshape(size,size).detach().numpy()
            
            plt.subplot(2,2,1)
            plt.imshow(u_pred)
            plt.colorbar()
            plt.subplot(2,2,2)
            plt.imshow(v_pred)
            plt.colorbar()
            plt.subplot(2,2,3)
            plt.imshow(p_pred)
            plt.colorbar()
            name_list.append(str(i))
            plt.savefig(os.path.join(path,name_list[j]))
            plt.show()
            j+=1
    else:
        if i % 100000 == 0:
            print(f'step: {i} loss = {loss.item()}')
            u_pred =net(inputs)[:,0].cpu().reshape(size,size).detach().numpy()
    
            v_pred =net(inputs)[:,1].cpu().reshape(size,size).detach().numpy()
    
            p_pred =net(inputs)[:,2].cpu().reshape(size,size).detach().numpy()
            
            plt.subplot(2,2,1)
            plt.imshow(u_pred)
            plt.colorbar()
            plt.subplot(2,2,2)
            plt.imshow(v_pred)
            plt.colorbar()
            plt.subplot(2,2,3)
            plt.imshow(p_pred)
            plt.colorbar()
            name_list.append(str(i))
            plt.savefig(os.path.join(path,name_list[j]))
            plt.show()
            j+=1
            
size_test=100   
x_test = torch.linspace(x_min, x_max, size_test).cuda()
y_test = torch.linspace(y_min, y_max, size_test).cuda()
x_test, y_test = torch.meshgrid(x_test, y_test)            
x_test = x_test.flatten()
y_test = y_test.flatten()
x_test = Variable(x_test, requires_grad=True)
y_test = Variable(y_test, requires_grad=True)

u_test =  np.pi *( torch.sin(np.pi * x_test)**2) * torch.sin(2 * np.pi * y_test)
v_test = -np.pi *( torch.sin(np.pi * y_test)**2) * torch.sin(2 * np.pi * x_test)
p_test = torch.sin(np.pi*x_test)*torch.sin(np.pi*y_test)

inputs_test = torch.stack((x_test, y_test), dim=1)
inputs_test = Variable(inputs_test , requires_grad = True)           
            
u_pred =net(inputs_test)[:,0].cpu().reshape(size_test,size_test).detach().numpy()
v_pred =net(inputs_test)[:,1].cpu().reshape(size_test,size_test).detach().numpy()
p_pred =net(inputs_test)[:,2].cpu().reshape(size_test,size_test).detach().numpy()

u_real = u_test.reshape(size_test,size_test).cpu().detach().numpy()
v_real = v_test.reshape(size_test,size_test).cpu().detach().numpy()
p_real = p_test.reshape(size_test,size_test).cpu().detach().numpy()



e_u =u_pred - u_real
e_v =v_pred - v_real
e_p =p_pred - p_real


end = time.time()


plt.subplot(3,2,1)
plt.imshow(u_pred)
plt.colorbar()
plt.subplot(3,2,2)
plt.imshow(u_real)
plt.colorbar()
plt.subplot(3,2,3)
plt.imshow(v_pred)
plt.colorbar()
plt.subplot(3,2,4)
plt.imshow(v_real)
plt.colorbar()
plt.subplot(3,2,5)
plt.imshow(p_pred)
plt.colorbar()
plt.subplot(3,2,6)
plt.imshow(p_real)
plt.colorbar()
plt.show()

plt.subplot(3,1,1)
plt.imshow(e_u)
plt.colorbar()
plt.subplot(3,1,2)
plt.imshow(e_v)
plt.colorbar()
plt.subplot(3,1,3)
plt.imshow(e_p)
plt.colorbar()
plt.show()
print("執行時間：%f 秒" % (end - start))
torch.save(net.state_dict(), 'model_.pt')
e_u=e_u.flatten()
e_v=e_v.flatten()
e_p= e_p.flatten()
# e_2norm = sum(e_u**2)
end = time.time()
u_2norm=0
for i in e_u: 
    u_2norm += i**2
u_2norm = math.sqrt(u_2norm/(size_test**2))

v_2norm=0
for i in e_v: 
    v_2norm += i**2
v_2norm = math.sqrt(v_2norm/(size_test**2))

p_2norm=0
for i in e_p: 
    p_2norm += i**2
p_2norm = math.sqrt(p_2norm/(size_test**2))

print('u_2norm: ',u_2norm)
print('v_2norm: ',v_2norm)
print('p_2norm: ',p_2norm)
print(name_list)








