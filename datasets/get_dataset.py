from datasets.tudl_db import TUDL_DB_Train, TUDL_DB_Test
from torch.utils.data import DataLoader

def GetDataset(option, db_nm, partition, batch_size, shuffle, drop_last, cls_nm=None, core_count=1):
    loader, db = None, None
    if db_nm == "tudl":
        if partition in ["train"]:
            db = TUDL_DB_Train(option, partition, cls_nm)
            loader = DataLoader(db, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=core_count, pin_memory=True)
        else:
            db = TUDL_DB_Test(option, partition, cls_nm)
            loader = DataLoader(db, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=core_count, pin_memory=True)
    return loader, db