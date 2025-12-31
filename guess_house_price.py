import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv("houses.csv")

X = data[["area", "rooms"]]
y = data["price"]

model = LinearRegression()
model.fit(X, y)

new_house = [[70, 3]]
prediction = model.predict(new_house)

print(f"Price - ≈ ${int(prediction[0])}")

plt.scatter(data["area"], y)
plt.xlabel("Area (m²)")
plt.ylabel("Price ($)")
plt.title("House price vs area")
plt.show()
