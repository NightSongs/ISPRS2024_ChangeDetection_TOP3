import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# 定义模型和优化器
model = torch.nn.Conv2d(3, 3, 3)
optimizer = torch.optim.SGD(model.parameters(), lr=5e-4)

# 定义学习率调度器
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2, eta_min=1e-5)

# 存储每个周期的学习率
learning_rates = []

# 运行多个周期并记录学习率
num_epochs = 253
for epoch in range(num_epochs):
    # 更新学习率
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    learning_rates.append(lr)

# 可视化学习率的变化
plt.plot(range(num_epochs), learning_rates)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('CosineAnnealingWarmRestarts Learning Rate Schedule')
plt.savefig('learning_rate_plot.png')