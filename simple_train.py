import torch
from accelerate import Accelerator
from diffusers.optimization import SchedulerType, get_scheduler
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor([index], dtype=torch.float32), torch.tensor([index + 1], dtype=torch.float32)


class MyDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset: torch.utils.data.Dataset, batch_size: int) -> None:
        super().__init__(dataset, batch_size=batch_size, shuffle=False)


model = Model()

loss_fn = MSELoss()
optimizer = AdamW(model.parameters(), lr=1e-5)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

dataset = MyDataset(100)
dataloader = MyDataLoader(dataset, batch_size=2)

accelerator = Accelerator(
    gradient_accumulation_steps=3,
)

optimizer, model, dataloader, scheduler =  accelerator.prepare(optimizer, model, dataloader, scheduler)
get_scheduler(
    SchedulerType.CONSTANT_WITH_WARMUP,
    optimizer,
    num_warmup_steps=20,
    num_training_steps=100,
)

for batch in dataloader:
    with accelerator.accumulate(model):
        inputs, targets = batch
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
