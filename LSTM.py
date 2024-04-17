import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split


candidate_admit_id = [] # เก็บ admit id ที่รอด exclude
vitalsigns_df = pd.DataFrame() # read ipd_vital_signs.csv -> sort จาก old ไป new
selected_cols = [] # ชื่อ feature (column) ที่เลือก เช่น BP, HR
n_feature = len(selected_cols)
label_col = 'is_dead' # ชื่อ column ที่เป็น label (0 or 1)
lookback = 12 # มองกลับไปกี่ timeslot
n_epochs = 100 # train กี่รอบ


class LSTM(nn.Module):
    def __init__(self, n_feature):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_feature, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
        # ค่อยปรับความ complex ของ model ทีหลัง

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x


def slide_window(features, label, lookback):
    _X = []
    _y = []

    for i in range(len(features)-lookback):
        _X.append(features[i:i+lookback])
        _y.append(label[i+lookback-1])

    return _X, _y


def create_dataset(vitalsigns, lookback):
    X = []
    y = []

    for admit_id, data_dict in vitalsigns.items():

        # assume ว่า label อยู่ column สุดท้าย
        features = data_dict['features']
        label = data_dict['label']

        _X, _y = slide_window(features, label, lookback)

        X.extend(_X)
        y.extend(_y)

    return X, y


def get_vitalsigns_data():
    vitalsigns = {}
    filtered_vitalsigns_df = vitalsigns_df[selected_cols + [label_col]]

    for admit_id in candidate_admit_id:
        vitalsigns[admit_id] = {}

        filtered_vitalsigns_df = vitalsigns_df[vitalsigns_df['admit_id']==admit_id]
        vitalsigns[admit_id]['features'] = filtered_vitalsigns_df[selected_cols]
        vitalsigns[admit_id]['label'] = filtered_vitalsigns_df[[label_col]]

    return vitalsigns


def main():
    vitalsigns = get_vitalsigns_data()
    X, y = create_dataset(vitalsigns, lookback)

    X_train, X_test, y_train, y_test = train_test_split(
                                            X, y, 
                                            test_size=0.3, 
                                            random_state=42, 
                                            shuffle=True
                                        )
    
    # init model
    model = LSTM()

    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    

    # training
    model.train()
    for epoch in range(n_epochs):
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}   Loss {loss:.3f}')

    
    # evaluate model
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        loss = loss_fn(y_pred, y_test)
        print(f'[Eval]: Loss {loss:.3f}')


main()