import tqdm
import torch
import torchmetrics
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class MovieRatingDataSet(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.id_cols = ['movieId', 'userId']
        self.categorical_cols = ['movieGenre1', 'movieGenre2', 'movieGenre3', 'userGenre1',
                                 'userGenre2', 'userGenre3', 'userGenre4', 'userGenre5']
        self.numerical_cols = ['releaseYear', 'movieRatingCount', 'movieAvgRating', 'movieRatingStddev',
                               'userRatingCount', 'userAvgRating', 'userRatingStddev']
        self.label_cols = ['label']
        self.genre_vocab = ['Film-Noir', 'Action', 'Adventure', 'Horror', 'Romance', 'War', 'Comedy', 'Western', 'Documentary',
                            'Sci-Fi', 'Drama', 'Thriller', 'Crime', 'Fantasy', 'Animation', 'IMAX', 'Mystery', 'Children', 'Musical']
        self.data = pd.read_csv(data_path, sep=',')
        for col in self.categorical_cols:
            self.data[col] = self.data[col].apply(lambda x: self.genre_vocab.index(x) if x in self.genre_vocab else len(self.genre_vocab))
        for col in self.id_cols:
            self.data[col] = self.data[col].astype('category').cat.codes
        self.emb_table_size = {
            'genre': len(self.genre_vocab),
            'movieId': self.data['movieId'].nunique(),
            'userId': self.data['userId'].nunique()
        }
        self.feature_num = {
            'sparse': len(self.categorical_cols) + len(self.id_cols),
            'dense': len(self.numerical_cols)
        }

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sparse_feature = self.data.iloc[idx][self.id_cols + self.categorical_cols].values
        dense_feature = self.data.iloc[idx][self.numerical_cols].values.astype(float)
        label = self.data.iloc[idx][self.label_cols].values.astype(float)
        sample = {
            'cate_features': torch.tensor(sparse_feature, dtype=torch.long),
            'num_features': torch.tensor(dense_feature, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.float)
        }
        return sample

class EmbeddingMLPModel(nn.Module):
    def __init__(self, emb_table_size, feature_num, emb_dim):
        super().__init__()
        self.genre_emb = nn.Embedding(emb_table_size['genre']+1, emb_dim)
        self.movie_id_emb = nn.Embedding(emb_table_size['movieId'], emb_dim)
        self.user_id_emb = nn.Embedding(emb_table_size['userId'], emb_dim)
        emb_size = sum(emb_table_size.values())
        self.feature_layer = nn.Linear(feature_num['sparse'] * emb_dim + feature_num['dense'], 128)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 1)
        self.output_layer = nn.Sigmoid()

    def forward(self, input):
        cate_features = input['cate_features']
        num_features = input['num_features']
        bottom_feature_movieid = self.movie_id_emb(cate_features[:, 0])
        bottom_feature_userid = self.user_id_emb(cate_features[:, 1])
        bottom_feature_genre = [self.genre_emb(cate_features[:, i+2]) for i in range(8)]
        x = torch.cat(bottom_feature_genre + [bottom_feature_movieid, bottom_feature_userid, num_features], dim=1)
        x = torch.relu(self.feature_layer(x))
        x = torch.relu(self.fc1(x))
        x = self.output_layer(self.fc2(x))
        return x

if __name__ == '__main__':
    train_dataset = MovieRatingDataSet('../../../../../../src/main/resources/webroot/sampledata/trainingSamples.csv')
    test_dataset = MovieRatingDataSet('../../../../../../src/main/resources/webroot/sampledata/testSamples.csv')
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=512)
    model = EmbeddingMLPModel(train_dataset.emb_table_size, train_dataset.feature_num, 10)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 度量指标
    metric_rocauc = torchmetrics.AUROC(task='binary')
    metric_prauc = torchmetrics.AveragePrecision(task='binary')
    metric_acc = torchmetrics.Accuracy(task='binary')
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        sample_count = 0
        with tqdm.tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", unit='batch') as pbar:
            for sample in train_dataloader:
                optimizer.zero_grad()
                outputs = model(sample)
                loss = criterion(outputs, sample['label'])
                # loss = torch.nn.functional.binary_cross_entropy(outputs, sample['label'])
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
                pbar.set_postfix({
                    'loss': epoch_loss / sample_count,
                    'acc': metric_acc.compute().item(),
                    'roc_auc': metric_rocauc.compute().item(),
                    'pr_auc': metric_prauc.compute().item()
                })
                pbar.update()
        print(
            f'Epoch {epoch + 1} done. loss:{epoch_loss / len(train_dataset)}, acc: {metric_acc.compute().item()}, roc_auc: {metric_rocauc.compute().item()}, pr_auc: {metric_prauc.compute().item()}')
        metric_rocauc.reset()
        metric_prauc.reset()
        metric_acc.reset()
    # 评估
    metric_rocauc.reset()
    metric_prauc.reset()
    metric_acc.reset()
    model.eval()
    with torch.no_grad():
        for sample in test_dataloader:
            outputs = model(sample)
            loss = criterion(outputs, sample['label'])
            metric_rocauc.update(outputs, sample['label'].to(torch.int))
            metric_prauc.update(outputs, sample['label'].to(torch.int))
            metric_acc.update(outputs, sample['label'].to(torch.int))
    print(f'Loss: {loss.item()}, ACC: {metric_acc.compute()}, ROC_AUC: {metric_rocauc.compute()}, PR_AUC: {metric_prauc.compute()}')


