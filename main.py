import numpy as np
import pandas as pd
import yfinance as yf
import talib as ta
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ---【重要】インポート修正箇所 ---
# VS Codeでエラーが出る場合、tensorflow.keras ではなく
# keras から直接インポートすると解決することが多いです
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
# ------------------------------

# ==============================================================================
# 0. 設定・定数 (CONFIG)
# ==============================================================================
TARGET_TICKER = '5243.T'  # 分析対象: カバー (グロース市場の注目銘柄)

# 相関関係のある外部データ
EXTRA_TICKERS = {
    '^GSPC': 'S&P500',    # 米国市場 (地合い)
    'JPY=X': 'USD_JPY',   # 為替
    '^VIX':  'VIX'        # 恐怖指数
}

START_DATE = '2010-01-01'
END_DATE = '2025-12-31'
PREDICTION_DAYS = 240     # 過去240日(約1年)の文脈を読む

# 特徴量リスト
BASE_FEATURES = ['Close', 'RSI', 'MACD', 'BB_UPPER', 'BB_LOWER', 'Log_Ret']
EXTRA_FEATURES = list(EXTRA_TICKERS.values())
FEATURE_COLS = BASE_FEATURES + EXTRA_FEATURES

# ==============================================================================
# 1. データ取得・結合関数
# ==============================================================================
def fetch_and_merge_data(target_ticker, extra_tickers):
    print(f"--- [1/5] データ取得開始: {target_ticker} ---")
    
    # メイン銘柄取得
    df = yf.download(target_ticker, start=START_DATE, end=END_DATE)
    
    # MultiIndex対応
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(target_ticker, axis=1, level=1)
        except:
            df.columns = df.columns.get_level_values(0)
            
    df_main = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # 外部データ結合
    for ticker, name in extra_tickers.items():
        print(f"   -> 外部データ結合中: {name} ({ticker})")
        extra_df = yf.download(ticker, start=START_DATE, end=END_DATE)
        
        if isinstance(extra_df.columns, pd.MultiIndex):
            try:
                vals = extra_df.xs(ticker, axis=1, level=1)['Close']
            except:
                vals = extra_df['Close']
        else:
            vals = extra_df['Close']
            
        df_main[name] = vals
        df_main[name] = df_main[name].ffill()

    df_main.dropna(inplace=True)
    return df_main

# ==============================================================================
# 2. 特徴量エンジニアリング
# ==============================================================================
def add_technical_indicators(df):
    print(f"--- [2/5] テクニカル指標計算 ---")
    data = df.copy()
    close = data['Close'].values
    
    # TA-Lib計算
    data['RSI'] = ta.RSI(close, timeperiod=14)
    macd, macdsignal, macdhist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    data['MACD'] = macd
    upper, middle, lower = ta.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    data['BB_UPPER'] = upper
    data['BB_LOWER'] = lower
    data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
    
    data.dropna(inplace=True)
    return data

# ==============================================================================
# 3. データ前処理
# ==============================================================================
def preprocess_data(data, prediction_days):
    dataset = data[FEATURE_COLS].values
    target = data['Close'].values.reshape(-1, 1)
    
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler_features.fit_transform(dataset)
    
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_target = scaler_target.fit_transform(target)
    
    x, y = [], []
    for i in range(prediction_days, len(scaled_data)):
        x.append(scaled_data[i-prediction_days:i])
        y.append(scaled_target[i, 0])
        
    x, y = np.array(x), np.array(y)
    return x, y, scaler_target, scaler_features

# ==============================================================================
# 4. モデル構築 (Bi-LSTM)
# ==============================================================================
def build_advanced_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))

    # 双方向LSTM
    model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(units=64, return_sequences=False)))
    model.add(Dropout(0.3))
    
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1)) 
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# ==============================================================================
# メイン処理
# ==============================================================================
if __name__ == "__main__":
    # 1. データ取得
    raw_df = fetch_and_merge_data(TARGET_TICKER, EXTRA_TICKERS)
    df = add_technical_indicators(raw_df)
    
    # データ分割
    split_ratio = 0.90
    split_index = int(len(df) * split_ratio)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index - PREDICTION_DAYS:]
    
    print(f"学習データ数: {len(train_df)}日分 / テストデータ数: {len(test_df) - PREDICTION_DAYS}日分")
    
    # 2. 学習
    x_train, y_train, scaler_target, scaler_features = preprocess_data(train_df, PREDICTION_DAYS)
    
    print(f"\n--- [3/5] AIモデル学習開始 (Target: {TARGET_TICKER}) ---")
    model = build_advanced_model((x_train.shape[1], x_train.shape[2]))
    
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    
    model.fit(x_train, y_train, epochs=50, batch_size=32, callbacks=[early_stop, reduce_lr])
    
    # 3. 検証・予測
    print(f"\n--- [4/5] 精度検証 & バックテスト ---")
    test_dataset = test_df[FEATURE_COLS].values
    test_scaled = scaler_features.transform(test_dataset)
    
    x_test = []
    for i in range(PREDICTION_DAYS, len(test_scaled)):
        x_test.append(test_scaled[i-PREDICTION_DAYS:i])
    x_test = np.array(x_test)
    
    predictions_scaled = model.predict(x_test)
    predictions = scaler_target.inverse_transform(predictions_scaled)
    actual_prices = test_df['Close'].values[PREDICTION_DAYS:]
    
    # グラフ表示
    plt.figure(figsize=(14, 7))
    plt.plot(actual_prices, color='#333333', label=f'Actual {TARGET_TICKER}', alpha=0.8)
    plt.plot(predictions, color='#00CC66', label='AI Prediction', linewidth=1.5)
    plt.title(f'{TARGET_TICKER} Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # バックテスト計算
    initial_balance = 1_000_000
    balance = initial_balance
    holdings = 0
    
    pred_flat = predictions.flatten()
    real_flat = actual_prices.flatten()
    
    for i in range(len(pred_flat) - 1):
        current_p = real_flat[i]
        next_pred_p = pred_flat[i+1]
        if next_pred_p > current_p * 1.005: # 0.5%上昇予測で買い
            if holdings == 0:
                holdings = balance / current_p
                balance = 0
        elif next_pred_p < current_p:
            if holdings > 0:
                balance = holdings * current_p
                holdings = 0
                
    final_price = real_flat[-1]
    if holdings > 0: balance = holdings * final_price
        
    print(f"AIトレード最終資産: {balance:,.0f} 円 (Buy&Hold比較: {initial_balance * (final_price/real_flat[0]):,.0f} 円)")
    
    # 4. 明日の予測とATR戦略
    print(f"\n--- [5/5] 明日のトレードプラン (AI + ATR) ---")
    last_sequence = test_df[FEATURE_COLS].tail(PREDICTION_DAYS)
    last_sequence_scaled = scaler_features.transform(last_sequence.values)
    X_future = np.array([last_sequence_scaled])
    
    pred_future_price = scaler_target.inverse_transform(model.predict(X_future))[0][0]
    
    # ATR計算
    high = raw_df['High'].values
    low = raw_df['Low'].values
    close_raw = raw_df['Close'].values
    atr = ta.ATR(high, low, close_raw, timeperiod=14)
    current_atr = atr[-1]
    current_price_today = raw_df['Close'].iloc[-1]
    
    print(f"現在値: {current_price_today:,.0f} 円 -> AI予測: {pred_future_price:,.0f} 円")
    print(f"推奨エントリー(指値): {current_price_today - (current_atr * 0.2):,.0f} 円 (押し目狙い)")
    print(f"推奨損切りライン: {current_price_today - (current_atr * 0.2) - (current_atr * 1.5):,.0f} 円")
    print("=" * 60)