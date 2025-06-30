import torch
import torchmetrics
import pandas as pd
import torch.nn as nn
import torch.utils.data as data_utils
from collections import OrderedDict

import tqdm
from torch.utils.data import DataLoader

# device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.set_default_device(device)
# print(f'Running on the device: {device}')

class MovieRatingDataset(data_utils.Dataset):
    def __init__(self, data_path):
        super().__init__()
        # 1、读取样本数据
        self.df_data = pd.read_csv(data_path, sep=',', on_bad_lines='skip').fillna('0')
        self.id_fea_cols = ['movieId', 'userId', 'userRatedMovie1']
        self.cat_fea_cols = ['movieGenre1', 'movieGenre2', 'movieGenre3', 'userGenre1', 'userGenre2', 'userGenre3',
                             'userGenre4', 'userGenre5']
        self.num_fea_cols = ['releaseYear', 'movieRatingCount', 'movieAvgRating', 'movieRatingStddev',
                             'userRatingCount', 'userAvgRating', 'userRatingStddev']
        self.genre_vocab = ['Film-Noir', 'Action', 'Adventure', 'Horror', 'Romance', 'War', 'Comedy', 'Western',
                            'Documentary',
                            'Sci-Fi', 'Drama', 'Thriller', 'Crime', 'Fantasy', 'Animation', 'IMAX', 'Mystery',
                            'Children', 'Musical']
        self.label_cols = ['label']
        # Min-Max 归一化（缩放到 [0, 1]）
        self.df_data[self.num_fea_cols] = (self.df_data[self.num_fea_cols] - self.df_data[self.num_fea_cols].min()) / (
                    self.df_data[self.num_fea_cols].max() - self.df_data[self.num_fea_cols].min())
        # 中文标签转索引编号
        for col in self.cat_fea_cols:
            self.df_data[col] = self.df_data[col].apply(
                lambda x: self.genre_vocab.index(x) if x in self.genre_vocab else len(self.genre_vocab))
        for col in self.id_fea_cols:
            self.df_data[col] = self.df_data[col].astype('category').cat.codes
        # 特征数量
        self.fea_size = {
            'cat_fea': len(self.id_fea_cols) + len(self.cat_fea_cols) - 1,  # 去掉wide侧独有的userRatedMovie1特征
            'num_fea': len(self.num_fea_cols)
        }
        # 特征取值大小
        id_fea_len = {col: self.df_data[col].nunique() for col in self.id_fea_cols}
        self.cat_fea_len = {
            'genre': len(self.genre_vocab),
            **id_fea_len
        }

    def __len__(self):
        return self.df_data.shape[0]

    def __getitem__(self, idx):
        genre_fea = self.df_data.iloc[idx][self.cat_fea_cols].values.astype(float)
        num_fea = self.df_data.iloc[idx][self.num_fea_cols].values.astype(float)
        movie_id_fea = self.df_data.iloc[idx]['movieId']
        user_id_fea = self.df_data.iloc[idx]['userId']
        rated_movie_fea = self.df_data.iloc[idx]['userRatedMovie1']
        label = self.df_data.iloc[idx][self.label_cols]
        sample = {
            'genre_fea': torch.tensor(genre_fea, dtype=torch.long),
            'num_fea': torch.tensor(num_fea, dtype=torch.float),
            'movie_id_fea': torch.tensor(movie_id_fea, dtype=torch.long),
            'user_id_fea': torch.tensor(user_id_fea, dtype=torch.long),
            'rated_movie_fea': torch.tensor(rated_movie_fea, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float)
        }
        return sample


class SimpleCrossedFeature(nn.Module):
    """简易交叉特征"""

    def __init__(self, cross_hash_size):
        super().__init__()
        if not cross_hash_size or cross_hash_size < 1:
            raise ValueError('cross_hash_size must be > 1. cross_hash_size: {}'.format(cross_hash_size))
        self.cross_hash_size = cross_hash_size

    def forward(self, features, radix):
        if not features or len(features) < 2:
            raise ValueError('features must be a list with length > 1. Given: {}'.format(features))
        if not radix:
            raise ValueError('radix must be larger than the max value of features. features:{}'.format(features))
        crossed = torch.zeros_like(features[0])
        for fea in features:
            crossed = crossed * radix + fea
        crossed = crossed % self.cross_hash_size
        return crossed


class WideNDeepModel(nn.Module):
    def __init__(self, cat_fea_len, emb_dim, cat_fea_size, num_fea_size, cross_hash_size):
        super().__init__()
        self.cat_fea_len = cat_fea_len
        self.cross_hash_size = cross_hash_size
        self.genre_emb = nn.Embedding(self.cat_fea_len['genre'] + 1, emb_dim)
        self.movie_id_emb = nn.Embedding(self.cat_fea_len['movieId'], emb_dim)
        self.user_id_emb = nn.Embedding(self.cat_fea_len['userId'], emb_dim)
        self.deep_nn = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(cat_fea_size * emb_dim + num_fea_size, 128)),
            ('act1', nn.ReLU()),
            ('fc2', nn.Linear(128, 128)),
            ('act2', nn.ReLU())
        ]))
        self.wide = nn.Embedding(self.cross_hash_size, 1)
        self.output = nn.Linear(128 + 1, 1)

    def forward(self, input):
        # emb
        genre_feas = [self.genre_emb(input['genre_fea'][:, i]) for i in range(8)]
        movie_id_fea = self.movie_id_emb(input['movie_id_fea'])
        user_id_fea = self.user_id_emb(input['user_id_fea'])

        # deep
        deep_fea = torch.concat(genre_feas + [movie_id_fea, user_id_fea, input['num_fea']],
                                dim=1)
        deep_out = self.deep_nn(deep_fea)
        cross = SimpleCrossedFeature(self.cross_hash_size)
        # wide
        wide_fea = cross([input['movie_id_fea'], input['rated_movie_fea']],
                         max(self.cat_fea_len['movieId'], self.cat_fea_len['userRatedMovie1']))
        wide_out = self.wide(wide_fea)
        x = self.output(torch.concat([wide_out, deep_out], dim=1))
        return torch.sigmoid(x)


if __name__ == '__main__':
    # generator = torch.Generator(device=device)
    dataset_train = MovieRatingDataset('../../../../../../src/main/resources/webroot/sampledata/trainingSamples.csv')
    dataset_test = MovieRatingDataset('../../../../../../src/main/resources/webroot/sampledata/testSamples.csv')
    train_dataloader = DataLoader(dataset_train, batch_size=512, shuffle=True)
    test_dataloader = DataLoader(dataset_test, batch_size=512, shuffle=False)
    # 实例化模型
    model = WideNDeepModel(cat_fea_len=dataset_train.cat_fea_len, emb_dim=10,
                           cat_fea_size=dataset_train.fea_size['cat_fea'],
                           num_fea_size=dataset_train.fea_size['num_fea'], cross_hash_size=10000)
    # 损失函数
    criterion = nn.BCELoss()
    # 梯度优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # metric
    metric_rocauc = torchmetrics.AUROC(task='binary')
    metric_prauc = torchmetrics.AveragePrecision(task='binary')
    metric_acc = torchmetrics.Accuracy(task='binary')
    #
    epochs_num = 5
    for epoch in range(epochs_num):
        model.train()
        epoch_loss = 0.0
        sample_count = 0
        with tqdm.tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{epochs_num}", unit='batch') as p_bar:
            for sample in train_dataloader:
                optimizer.zero_grad()
                outputs = model(sample)
                loss = criterion(outputs, sample['label'])
                loss.backward()
                optimizer.step()
                # 累积loss
                sample_count += sample['label'].shape[0]
                epoch_loss += loss.item() * sample['label'].shape[0]
                # 指标更新
                if 0 < sample['label'].sum() < sample['label'].shape[0]:
                    metric_acc.update(outputs, sample['label'].to(torch.int))
                    metric_rocauc.update(outputs, sample['label'].to(torch.int))
                    metric_prauc.update(outputs, sample['label'].to(torch.int))
                # 指标打印
                p_bar.set_postfix({
                    'loss': epoch_loss / sample_count,
                    'acc': metric_acc.compute().item(),
                    'roc_auc': metric_rocauc.compute().item(),
                    'pr_auc': metric_prauc.compute().item()
                })
                p_bar.update()
        print(
            f'Epoch {epoch + 1} done. loss:{epoch_loss / len(dataset_train)}, acc: {metric_acc.compute().item()}, roc_auc: {metric_rocauc.compute().item()}, pr_auc: {metric_prauc.compute().item()}')
        metric_acc.reset()
        metric_rocauc.reset()
        metric_prauc.reset()
    # eval
    metric_rocauc.reset()
    metric_prauc.reset()
    metric_acc.reset()
    model.eval()
    with torch.no_grad():
        for sample in test_dataloader:
            outputs = model(sample)
            loss = criterion(outputs, sample['label'])
            metric_acc.update(outputs, sample['label'].to(torch.int))
            metric_rocauc.update(outputs, sample['label'].to(torch.int))
            metric_prauc.update(outputs, sample['label'].to(torch.int))
    print(
        f'Loss: {loss.item()}, ACC: {metric_acc.compute()}, ROC_AUC: {metric_rocauc.compute()}, PR_AUC: {metric_prauc.compute()}')
