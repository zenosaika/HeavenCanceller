# HeavenCanceller

### Basic Pytorch
```
# ค่าด้านใน link กัน
n = np.array([1, 2, 3])
t = torch.from_numpy(n)
n2 = t.numpy()

# กลายเป็นคนละตัวกันเลย แก้ค่าได้ไม่กระทบเพื่อน
n = np.array([1, 2, 3])
t = torch.Tensor(n)
n2 = np.array(t.data)
```
- Tensor + - * / เป็น element-wise ถ้าจะคูณ matrix ใช้ torch.matmal(t1, t2)
- t.mean(), t.sum(), t.std(), t.max(), t.min()
- change shape ใช้ t.view(x, y), t.flatten()
- timemachines
    - tbats (Trigonometric Seasonal, Box-Cox Transformation, ARIMA Errors, Trend, and Seasonal Components)
    - prophet (Facebook's time series forecasting model)
- A rolling-forecast scenario will be used, also called walk-forward model validation.