from src.data_loader import prepare_data
from src.train import train_nssm
from src.evaluate import evaluate_model

train, val, test = prepare_data()
model = train_nssm(train)
rmse, mae = evaluate_model(model, test)

print("NSSM Results")
print("RMSE:", rmse)
print("MAE:", mae)
