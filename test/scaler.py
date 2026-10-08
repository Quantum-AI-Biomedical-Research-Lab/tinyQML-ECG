import numpy as np
from src.scaler.standard_scaler import ZScoreScaler, ParetoScaler, VASTScaler
from src.scaler.min_max_scaler import MinMaxScaler
from src.scaler.max_absolute_scaler import MaxAbsoluteScaler
from src.scaler.robust_scaler import RobustScaler

x = np.array([
    [-0.178, 0.469, -0.180, 930, 120],
    [-0.152, 0.521, -0.157, 950, 100],
    [-0.200, 0.500, -0.190, 900, 110],
])

def main():
    scaler = ZScoreScaler()
    zscore = scaler.fit_transform(x)
    print("ZScore")
    print(zscore)

    scaler = ParetoScaler()
    pareto = scaler.fit_transform(x)
    print("Pareto")
    print(pareto)
    
    scaler = VASTScaler()
    vast = scaler.fit_transform(x)
    print("VAST")
    print(vast)

    scaler = VASTScaler()
    vast = scaler.fit_transform(x)
    print("VAST")
    print(vast)

    scaler = MinMaxScaler()
    mm = scaler.fit_transform(x)
    print("MinMax")
    print(mm)

    scaler = MaxAbsoluteScaler()
    ma = scaler.fit_transform(x)
    print("MaxAbsolute")
    print(ma)

    scaler = RobustScaler()
    rbs = scaler.fit_transform(x)
    print("Robust")
    print(rbs)

if __name__ == "__main__":
    main()