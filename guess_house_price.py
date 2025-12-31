import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Загружаем данные
data = pd.read_csv("houses.csv")

# 2. Признаки и цель
X = data[["area", "rooms"]]
y = data["price"]

# 3. Модель
model = LinearRegression()
model.fit(X, y)

# 4. Новые данные
new_house = [[70, 3]]
prediction = model.predict(new_house)

print(f"Цена дома ≈ ${int(prediction[0])}")

# 5. График (площадь vs цена)
plt.scatter(data["area"], y)
plt.xlabel("Площадь (м²)")
plt.ylabel("Цена ($)")
plt.title("Цена дома vs площадь")
plt.show()