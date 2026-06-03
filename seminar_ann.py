# =====================================================
# НЕЙРОННЫЕ СЕТИ ДЛЯ ПРОГНОЗИРОВАНИЯ ЭНЕРГОПОТРЕБЛЕНИЯ
# =====================================================

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from tensorflow.keras import layers, optimizers, callbacks

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

# ==================== НАСТРОЙКИ ====================
data_dir = 'data/'
file_name = 'RU_Electricity_Market_PZ_dayahead_price_volume.csv'


CUSTOM_DTYPES = {
    'price_eur': np.float32,
    'price_sib': np.float32,
    'consumption_eur': np.float32,
    'consumption_sib': np.float32
}

# ==================== ЗАГРУЗКА ДАННЫХ ====================
source_data = pd.read_csv(
    data_dir + file_name,
    parse_dates=['timestep'],
    dtype=CUSTOM_DTYPES
)
source_data.index = source_data['timestep']
source_data.drop(columns=['timestep'], inplace=True)

target_name = 'consumption_eur'
print(f"Данные загружены: {len(source_data)} записей")
print(f"Период: {source_data.index.min()} — {source_data.index.max()}")


# ==================== FEATURE ENGINEERING ====================
data = pd.DataFrame()
data[target_name] = source_data[target_name]
data['hour'] = data.index.hour.values
data['dayofweek'] = data.index.dayofweek.values
data['weekofyear'] = data.index.isocalendar().week.astype(int)
data['month'] = data.index.month.values
data['year'] = data.index.year.values
data['weekend'] = data.index.dayofweek.isin([5, 6]).astype(int)  # суббота/воскресенье
data['lunch'] = data.index.hour.isin([12, 13, 14]).astype(int)

# One-hot encoding
data = pd.get_dummies(data, columns=['hour'], prefix='hour')

# Синус-косинус для сезонности
for w in range(1, 4):
    data[f'sin_{w}'] = np.sin(2 * np.pi * data.index.dayofyear / 365 * w)
    data[f'cos_{w}'] = np.cos(2 * np.pi * data.index.dayofyear / 365 * w)

# Лаги (ключевой момент для прогнозирования!)
lag_settings = [24, 48, 72, 96, 120, 144, 168]  # добавил 168 (неделя)
for shift in lag_settings:
    data[f'lag_{shift}'] = data[target_name].shift(shift)

# Удаляем строки с NaN
data.dropna(inplace=True)

# Разделение на признаки и цель
predictors_name = [col for col in data.columns if col != target_name]
print(f"\nПризнаков: {len(predictors_name)}")
print(f"Признаки: {predictors_name[:5]}...")

# ==================== МАСШТАБИРОВАНИЕ ====================
# ВАЖНО: масштабируем после разделения train/val, но пока что сохраняем параметры
scaler_mean = data.mean(axis=0)
scaler_std = data.values.std(axis=0)
data_scaled = (data.values - scaler_mean) / scaler_std

predictors = data_scaled[:, data.columns.isin(predictors_name)]
target = data_scaled[:, data.columns.isin([target_name])].flatten()

# ==================== ПОДГОТОВКА SEQUENCES ====================
horizon = 24  # прогноз на 24 часа
batch_size = 32
network_epochs = 3
learning_rate = 0.001

def create_sequences(predictors, target, horizon):
    """Создает последовательности для обучения"""
    X, y = [], []
    for i in range(len(predictors) - horizon):
        X.append(predictors[i:i+horizon])
        y.append(target[i+horizon])
    return np.array(X), np.array(y)

X, y = create_sequences(predictors, target, horizon)
print(f"\nФорма X: {X.shape}")  # (samples, horizon, features)
print(f"Форма y: {y.shape}")    # (samples,)

# Разделение на train/val (80/20)
split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# ==================== ПОСТРОЕНИЕ МОДЕЛИ ====================
# Вариант 1: MLP (как в исходном коде)
def build_mlp(input_shape):
    model = keras.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # прогноз на 1 шаг
    ])
    return model

# Вариант 2: LSTM (рекомендуется показать на семинаре)
def build_lstm(input_shape):
    model = keras.Sequential([
        layers.LSTM(64, input_shape=input_shape, return_sequences=True),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    return model

# Выбираем модель
model = build_mlp(X_train.shape[1:])
# model = build_lstm(X_train.shape[1:])  # раскомментировать для LSTM

model.summary()

# ==================== КОМПИЛЯЦИЯ ====================
model.compile(
    optimizer=optimizers.Adam(learning_rate=learning_rate),
    loss='mae',
    metrics=['mae', 'mse']
)

# ==================== CALLBACKS ====================
callbacks_list = [
    callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
    callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, verbose=0)
]

# ==================== ОБУЧЕНИЕ ====================
print("\n" + "="*50)
print("НАЧАЛО ОБУЧЕНИЯ")
print("="*50)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=network_epochs,
    batch_size=batch_size,
    # callbacks=callbacks_list,
    verbose=1
)

# ==================== ВИЗУАЛИЗАЦИЯ ОБУЧЕНИЯ ====================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('Loss (MAE)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MAE')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['mae'], label='Train MAE')
axes[1].plot(history.history['val_mae'], label='Val MAE')
axes[1].set_title('Mean Absolute Error')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# ==================== ПРОГНОЗИРОВАНИЕ ====================
# Прогноз на валидационной выборке
y_pred = model.predict(X_val)

# Обратное масштабирование
target_mean = scaler_mean[data.columns.get_loc(target_name)]
target_std = scaler_std[data.columns.get_loc(target_name)]

y_val_original = y_val * target_std + target_mean
y_pred_original = y_pred.flatten() * target_std + target_mean

# ==================== МЕТРИКИ ====================
mae = np.mean(np.abs(y_val_original - y_pred_original))
mape = np.mean(np.abs((y_val_original - y_pred_original) / y_val_original)) * 100

print(f"\n{'='*50}")
print(f"РЕЗУЛЬТАТЫ НА ВАЛИДАЦИИ")
print(f"{'='*50}")
print(f"MAE:  {mae:.2f} MWh")
print(f"MAPE: {mape:.2f}%")
print(f"{'='*50}")

# ==================== ВИЗУАЛИЗАЦИЯ ПРОГНОЗА ====================
# Покажем первые 500 часов
n_show = min(500, len(y_val_original))

plt.figure(figsize=(15, 6))
plt.plot(y_val_original[:n_show], label='Actual', alpha=0.7)
plt.plot(y_pred_original[:n_show], label='Predicted', alpha=0.7)
plt.title(f'Прогноз энергопотребления (MAE = {mae:.2f} MWh, MAPE = {mape:.1f}%)')
plt.xlabel('Часы')
plt.ylabel('MWh')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ==================== СОХРАНЕНИЕ МОДЕЛИ И СКАЛЕРОВ ====================
# Сохраняем модель
model.save('electricity_forecast_model.keras')
print("\n✓ Модель сохранена: electricity_forecast_model.keras")

# Сохраняем скалеры
scaler_info = {
    'target_name': target_name,
    'target_mean': float(target_mean),
    'target_std': float(target_std),
    'feature_names': predictors_name,
    'horizon': horizon
}

with open('scaler_info.json', 'w') as f:
    json.dump(scaler_info, f, indent=2)
print("✓ Скалеры сохранены: scaler_info.json")

# ==================== ПРИМЕР ИСПОЛЬЗОВАНИЯ ====================
print("\n" + "="*50)
print("ПРИМЕР ЗАГРУЗКИ МОДЕЛИ (в продакшне)")
print("="*50)
print("""
# Загрузка модели
loaded_model = keras.models.load_model('electricity_forecast_model.keras')

# Загрузка скалеров
with open('scaler_info.json', 'r') as f:
    scaler_info = json.load(f)

# Прогноз для новых данных
# new_data_scaled = (new_data - scaler_mean) / scaler_std
# prediction_scaled = loaded_model.predict(new_data_scaled)
# prediction = prediction_scaled * scaler_info['target_std'] + scaler_info['target_mean']
""")