import torch
import torchmetrics
import pandas as pd
import torch.utils.data as data_utils
import torch.nn as nn
import tqdm
from torchmetrics.utilities import dim_zero_sum


class MovieRatingDataset(data_utils.Dataset):
    def __init__(self, datapath):
        self.df_data = pd.read_csv(datapath, sep=',', on_bad_lines='skip').fillna('0')
        self.id_fea_cols = ['movieId', 'userId']
        self.cat_fea_cols = ['movieGenre1', 'userGenre1']
        self.con_fea_cols = ['releaseYear', 'movieRatingCount', 'movieAvgRating', 'movieRatingStddev',
                             'userRatingCount', 'userAvgRating', 'userRatingStddev']
        self.genre_vocab = ['Film-Noir', 'Action', 'Adventure', 'Horror', 'Romance', 'War', 'Comedy', 'Western',
                            'Documentary',
                            'Sci-Fi', 'Drama', 'Thriller', 'Crime', 'Fantasy', 'Animation', 'IMAX', 'Mystery',
                            'Children', 'Musical']
        self.label_cols = ['label']
        # id特征编码
        for col in self.id_fea_cols:
            self.df_data[col] = self.df_data[col].astype('category').cat.codes
        # 类别特征编码
        for col in self.cat_fea_cols:
            self.df_data[col] = self.df_data[col].apply(
                lambda x: self.genre_vocab.index(x) if x in self.genre_vocab else len(self.genre_vocab))
        # 数值特征归一化
        for col in self.con_fea_cols:
            self.df_data[col] = (self.df_data[col] - self.df_data[col].min()) / (
                    self.df_data[col].max() - self.df_data[col].min())
        # 特征数量
        self.num_cat_features = {
            **{col: self.df_data[col].nunique() for col in self.id_fea_cols},
            **{col: len(self.genre_vocab) for col in self.cat_fea_cols}
        }
        self.num_con_features = len(self.con_fea_cols)

    def __len__(self):
        return self.df_data.shape[0]

    def __getitem__(self, idx):
        cat_fea = {
            **{name: self.df_data.iloc[idx][name] for name in self.id_fea_cols},
            **{name: self.df_data.iloc[idx][name] for name in self.cat_fea_cols}
        }
        num_fea = self.df_data.iloc[idx][self.con_fea_cols]
        label = self.df_data.iloc[idx][self.label_cols]
        sample = {
            'cat_fea': {name: torch.tensor(fea, dtype=torch.long) for name, fea in cat_fea.items()},
            'num_fea': torch.tensor(num_fea, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.float)
        }
        return sample


class DeepFMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 一阶稀疏特征
        self.first_order_emb = nn.ModuleDict({
            name: nn.Embedding(num_embeddings+1, 1) for name, num_embeddings in config['num_cat_features'].items()
        })
        self.fm_1st_order_output = nn.Linear(len(config['num_cat_features']) + config['num_con_features'], 1)
        # 二阶稀疏特征
        embedding_dim = config.get('embedding_dim', 10)
        self.second_order_emb = nn.ModuleDict({
            name: nn.Embedding(num_embeddings+1, embedding_dim) for name, num_embeddings in
            config['num_cat_features'].items()
        })
        # DNN
        hidden_units = config['hidden_units']
        deep_layers = [
            nn.Linear(len(config['num_cat_features']) * embedding_dim + config['num_con_features'], hidden_units[0]),
            nn.BatchNorm1d(hidden_units[0]),
            nn.ReLU(),
            nn.Dropout(config.get('dropout_rate', 0.3))
        ]
        for i in range(1, len(hidden_units)):
            deep_layers.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))
            deep_layers.append(nn.BatchNorm1d(hidden_units[i]))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(config.get('dropout_rate', 0.3)))
        self.deep_nn = nn.Sequential(*deep_layers)
        self.deep_output = nn.Linear(hidden_units[-1], 1)
        self._init_weight()

    def _init_weight(self):
        # embeddings
        for emb_layer in self.first_order_emb.values():
            nn.init.uniform_(emb_layer.weight)
        for emb_layer in self.second_order_emb.values():
            nn.init.xavier_uniform_(emb_layer.weight)
        nn.init.xavier_normal_(self.fm_1st_order_output.weight)
        nn.init.constant_(self.fm_1st_order_output.bias, 0)
        for layer in self.deep_nn:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)
        nn.init.xavier_normal_(self.deep_output.weight)
        nn.init.constant_(self.deep_output.bias, 0)
        # nn
        pass

    def forward(self, inputs):
        # FM一阶特征交叉
        fm_1st_order_fea = torch.concat(
            [self.first_order_emb[name](fea) for name, fea in inputs['cat_fea'].items()] + [inputs['num_fea']], dim=1)
        fm_1st_order_output = self.fm_1st_order_output(fm_1st_order_fea)
        # FM二阶特征交叉
        fm_2nd_order_fea = [self.second_order_emb[name](fea) for name, fea in inputs['cat_fea'].items()]
        fm_2nd_order_sum = torch.sum(torch.stack(fm_2nd_order_fea), dim=0)
        fm_2nd_order_sum_square = torch.sum(fm_2nd_order_sum ** 2, dim=1, keepdim=True)
        fm_2nd_order_square_sum = torch.sum(torch.stack(fm_2nd_order_fea) ** 2, dim=(0, 2), keepdim=True).squeeze(0)
        fm_2nd_order_output = 0.5 * (fm_2nd_order_sum_square - fm_2nd_order_square_sum)
        # Deep
        deep_fea = torch.concat(fm_2nd_order_fea + [inputs['num_fea']], dim=1)
        deep_output = self.deep_output(self.deep_nn(deep_fea))
        # Add
        combined = fm_1st_order_output + fm_2nd_order_output + deep_output

        return torch.sigmoid(combined)


if __name__ == '__main__':
    # 数据集及样本
    dataset_train = MovieRatingDataset('../../../../../../src/main/resources/webroot/sampledata/trainingSamples.csv')
    dataset_test = MovieRatingDataset('../../../../../../src/main/resources/webroot/sampledata/testSamples.csv')
    dataloader_train = data_utils.DataLoader(dataset_train, batch_size=12, shuffle=True)
    dataloader_test = data_utils.DataLoader(dataset_test, batch_size=12, shuffle=False)
    # 模型
    model_config = {
        'num_cat_features': dataset_train.num_cat_features,
        'num_con_features': dataset_train.num_con_features,
        'hidden_units': [32, 16, 8],
        'embedding_dim': 10,
        'dropout_rate': 0.3
    }
    model = DeepFMModel(config=model_config)
    # loss
    criterion = nn.BCELoss()
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    # 指标
    metric_acc = torchmetrics.Accuracy(task='binary')
    metric_rocauc = torchmetrics.AUROC(task='binary')
    metric_prauc = torchmetrics.AveragePrecision(task='binary')
    # 迭代训练
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        sample_count = 0
        with tqdm.tqdm(total=len(dataloader_train), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as p_bar:
            for sample in dataloader_train:
                model.zero_grad()
                labels = sample['label']
                outputs = model(sample)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(labels)
                sample_count += len(labels)
                # 更新指标，正负样本都存在时
                if 0 < labels.sum() < labels.shape[0]:
                    metric_acc.update(outputs, labels.to(torch.int))
                    metric_prauc.update(outputs, labels.to(torch.int))
                    metric_rocauc.update(outputs, labels.to(torch.int))
                # 更新进度条
                p_bar.set_postfix({
                    'loss': epoch_loss / sample_count,
                    'acc': metric_acc.compute().item(),
                    'roc_auc': metric_rocauc.compute().item(),
                    'pr_auc': metric_prauc.compute().item()
                })
                p_bar.update()
        print(
            f'Epoch {epoch + 1} done. loss: {epoch_loss / sample_count}, acc: {metric_acc.compute().item()}, roc_auc: {metric_rocauc.compute().item()}, pr_auc: {metric_prauc.compute().item()}')
        metric_acc.reset()
        metric_rocauc.reset()
        metric_prauc.reset()
    # 评估
    model.eval()
    avg_loss = 0.0
    eval_count = 0
    with torch.no_grad():
        for sample in tqdm.tqdm(dataloader_test):
            labels = sample['label']
            outputs = model(sample)
            loss = criterion(outputs, labels)
            avg_loss += loss.item() * len(labels)
            eval_count += len(labels)
            metric_acc.update(outputs, labels.to(torch.int))
            metric_rocauc.update(outputs, labels.to(torch.int))
            metric_prauc.update(outputs, labels.to(torch.int))
    print(
        f'Loss: {avg_loss / eval_count}, ACC: {metric_acc.compute()}, ROC_AUC: {metric_rocauc.compute()}, PR_AUC: {metric_prauc.compute()}')
