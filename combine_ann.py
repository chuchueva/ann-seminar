from tensorflow.keras import layers, Model

feature_list = ['x']

# --- ВЕТКА 1: Обработка временных рядов (История потребления) ---
# На вход подаем историю за 24 часа с 1 признаком (мощность) -> shape=(24, 1)
history_input = layers.Input(shape=(24, len(feature_list)), name="energy_history")
x1 = layers.LSTM(64, return_sequences=False)(history_input)
x1 = layers.Dropout(0.2)(x1)

# --- ВЕТКА 2: Обработка статических данных (Контекст) ---
# На вход подаем 5 признаков (температура, влажность, день недели и т.д.) -> shape=(5,)
context_input = layers.Input(shape=(5,), name="weather_and_date")
x2 = layers.Dense(32, activation="relu")(context_input)

# --- ОБЪЕДИНЕНИЕ ВЕТОК ---
# Соединяем выходы обеих веток в один вектор
combined = layers.Concatenate()([x1, x2])

# --- ФИНАЛЬНЫЕ СЛОИ (Прогноз) ---
x3 = layers.Dense(32, activation="relu")(combined)
# Выходной слой: 1 нейрон без активации, так как предсказываем конкретное число (кВт)
output = layers.Dense(1, name="energy_prediction")(x3)

# --- СБОРКА МОДЕЛИ ---
model = Model(inputs=[history_input, context_input], outputs=output)

# Компиляция для задачи регрессии
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Вывод структуры модели на экран
model.summary()
