import time
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset

start_time = time.time()

# 你的代码
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=4, shuffle=True)
for i, item in enumerate(dataloader):
    pass

end_time = time.time()

print(f"运行时间: {end_time - start_time:.2f} 秒")
