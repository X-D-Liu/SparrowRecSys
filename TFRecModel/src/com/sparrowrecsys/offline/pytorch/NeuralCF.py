from typing import Literal

import torch
import pandas as pd
import torch.nn as nn
import torch.utils.data as data_utils
import torchmetrics
import tqdm


class MovieRatingDataset(data_utils.Dataset):
    def __init__(self, datapath):
        super().__init__()
        self.df_data = pd.read_csv(datapath, sep=',', on_bad_lines='skip').fillna('0')
        self.id_cols = ['movieId', 'userId']
        self.label_cols = ['label']
        for col in self.id_cols:
            self.df_data[col] = self.df_data[col].astype('category').cat.codes
        self.fea_len = {
            col: self.df_data[col].nunique() for col in self.id_cols
        }

    def __len__(self):
        return self.df_data.shape[0]

    def __getitem__(self, idx):
        movie_id_fea = self.df_data.iloc[idx]['movieId']
        user_id_fea = self.df_data.iloc[idx]['userId']
        labels = self.df_data.iloc[idx][self.label_cols]
        sample = {
            'movie_id_fea': torch.tensor(movie_id_fea, dtype=torch.long),
            'user_id_fea': torch.tensor(user_id_fea, dtype=torch.long),
            'label': torch.tensor(labels, dtype=torch.float)
        }
        return sample


class NeuralCFModel(nn.Module):
    def __init__(self, fea_len, emb_dim, hidden_units, style: Literal['neural_cf', 'two_tower']):
        super().__init__()
        if style not in {'neural_cf', 'two_tower'}:
            raise ValueError("style can only be 'neural_cf' or 'two_tower'")
        self.style = style
        self.emb_dim = emb_dim
        self.movie_id_emb = nn.Embedding(fea_len['movieId'], self.emb_dim)
        self.user_id_emb = nn.Embedding(fea_len['userId'], self.emb_dim)
        if self.style == 'neural_cf':
            # 直接拼接
            interact_layers = [
                nn.Linear(self.emb_dim * 2, hidden_units[0]),
                nn.ReLU()
            ]
            for i in range(1, len(hidden_units)):
                interact_layers.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))
                interact_layers.append(nn.ReLU())
            self.interact_tower = nn.Sequential(*interact_layers)
            self.output_layer = nn.Linear(hidden_units[-1], 1)
        elif self.style == 'two_tower':
            # user塔
            user_layers = [
                nn.Linear(self.emb_dim, hidden_units[0]),
                nn.ReLU()
            ]
            for i in range(1, len(hidden_units)):
                user_layers.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))
                user_layers.append(nn.ReLU())
            self.user_tower = nn.Sequential(*user_layers)
            # item塔
            item_layers = [
                nn.Linear(self.emb_dim, hidden_units[0]),
                nn.ReLU()
            ]
            for i in range(1, len(hidden_units)):
                item_layers.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))
                item_layers.append(nn.ReLU())
            self.item_tower = nn.Sequential(*item_layers)

    def forward(self, inputs):
        movie_id_emb = self.movie_id_emb(inputs['movie_id_fea'])
        user_id_emb = self.user_id_emb(inputs['user_id_fea'])
        x = None
        if self.style == 'neural_cf':
            x = torch.concat([movie_id_emb, user_id_emb], dim=1)
            x = self.interact_tower(x)
            x = self.output_layer(x)
        elif self.style == 'two_tower':
            movie_out = self.item_tower(movie_id_emb)
            user_out = self.user_tower(user_id_emb)
            x = torch.einsum('bi,bi->b', movie_out, user_out).unsqueeze(1)
        return torch.sigmoid(x)


if __name__ == '__main__':
    dataset_train = MovieRatingDataset('../../../../../../src/main/resources/webroot/sampledata/trainingSamples.csv')
    dataset_test = MovieRatingDataset('../../../../../../src/main/resources/webroot/sampledata/testSamples.csv')
    train_dataloader = data_utils.DataLoader(dataset_train, batch_size=512, shuffle=True)
    test_dataloader = data_utils.DataLoader(dataset_test, batch_size=512)
    # 模型
    model = NeuralCFModel(fea_len=dataset_train.fea_len, emb_dim=10, hidden_units=[10, 10], style='neural_cf')
    # 损失函数
    criterion = nn.BCELoss()
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 指标
    metric_acc = torchmetrics.Accuracy(task='binary')
    metric_rocauc = torchmetrics.AUROC(task='binary')
    metric_prauc = torchmetrics.AveragePrecision(task='binary')
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        sample_count = 0
        with tqdm.tqdm(total=len(train_dataloader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as p_bar:
            for sample in train_dataloader:
                optimizer.zero_grad()
                outputs = model(sample)
                loss = criterion(outputs, sample['label'])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * sample['label'].shape[0]
                sample_count += sample['label'].shape[0]
                # 指标更新
                if 0 < sample['label'].sum() < sample['label'].shape[0]:
                    metric_acc.update(outputs, sample['label'].to(torch.int))
                    metric_rocauc.update(outputs, sample['label'].to(torch.int))
                    metric_prauc.update(outputs, sample['label'].to(torch.int))
                # 进度条更新
                p_bar.set_postfix({
                    'loss': epoch_loss / sample_count,
                    'acc': metric_acc.compute().item(),
                    'roc_auc': metric_rocauc.compute().item(),
                    'pr_auc': metric_prauc.compute().item()
                })
                p_bar.update()
        print(f'Epoch {epoch + 1} done. loss:{epoch_loss / len(dataset_train)}, acc: {metric_acc.compute().item()}, roc_auc: {metric_rocauc.compute().item()}, pr_auc: {metric_prauc.compute().item()}')
        metric_acc.reset()
        metric_rocauc.reset()
        metric_prauc.reset()
    # eval
    metric_rocauc.reset()
    metric_prauc.reset()
    metric_acc.reset()
    model.eval()
    avg_loss = 0.0
    with torch.no_grad():
        for sample in test_dataloader:
            outputs = model(sample)
            loss = criterion(outputs, sample['label'])
            avg_loss += loss.item() * sample['label'].shape[0]
            metric_acc.update(outputs, sample['label'].to(torch.int))
            metric_rocauc.update(outputs, sample['label'].to(torch.int))
            metric_prauc.update(outputs, sample['label'].to(torch.int))
    print(
        f'Loss: {avg_loss / len(test_dataloader.dataset)}, ACC: {metric_acc.compute()}, ROC_AUC: {metric_rocauc.compute()}, PR_AUC: {metric_prauc.compute()}')