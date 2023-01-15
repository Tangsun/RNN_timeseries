import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp

class TimeseriesDataset(Dataset):
    def __init__(self, mat_file, seq_len, fut_len):
        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']
        self.n = seq_len
        self.f = fut_len

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        dsId = self.D[idx, 0].astype(int)-1
        t = self.D[idx, 1].astype(int)
        if t < self.n:
            X = torch.from_numpy(self.T[dsId, 0][0: self.n]).float()
            y = torch.from_numpy(self.T[dsId, 0][self.n: self.n+self.f]).float()
        elif t > self.T[dsId][0].shape[0] - self.f:
            t_st = self.T[dsId][0].shape[0] - self.n - self.f
            t_ed = self.T[dsId][0].shape[0] - self.f
            X = torch.from_numpy(self.T[dsId, 0][t_st: t_ed]).float()
            y = torch.from_numpy(self.T[dsId, 0][t_ed: t_ed+self.f]).float()
        else:
            t_st = t - self.n
            X = torch.from_numpy(self.T[dsId, 0][t_st: t]).float()
            y = torch.from_numpy(self.T[dsId, 0][t: t+self.f]).float()

        return X, y