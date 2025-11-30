import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import talib as ta
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# TensorFlow / Keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

# ==============================================================================
# 0. アプリ設定
# ==============================================================================
st.set_page_config(page_title="AIトレード分析Pro", layout="wide")

st.title("📈 機関投資家仕様 AIトレード分析 Pro")
st.markdown("""
**フル機能**: AI予測 & バックテスト | 水平線 | フィボナッチ | VWAP | ADX | カルマンフィルタ | ATR | 高度パターン認識
""")

# ==============================================================================
# 1. 計算ロジック
# ==============================================================================

# カルマンフィルタ
def apply_kalman_filter(prices, Q=1e-5, R=0.01):
    n_iter = len(prices)
    sz = (n_iter,) 
    xhat = np.zeros(sz); P = np.zeros(sz)
    xhatminus = np.zeros(sz); Pminus = np.zeros(sz); K = np.zeros(sz)
    xhat[0] = prices[0]; P[0] = 1.0
    for k in range(1, n_iter):
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + Q
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (prices[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
    return xhat

# データ取得
@st.cache_data(ttl=3600)
def fetch_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        try: df = df.xs(ticker, axis=1, level=1)
        except: df.columns = df.columns.get_level_values(0)
    
    df_main = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    extra_tickers = {'^GSPC': 'S&P500', 'JPY=X': 'USD_JPY', '^VIX': 'VIX'}
    for t, name in extra_tickers.items():
        extra = yf.download(t, start=start_date, end=end_date)
        if isinstance(extra.columns, pd.MultiIndex):
            try: vals = extra.xs(t, axis=1, level=1)['Close']
            except: vals = extra['Close']
        else: vals = extra['Close']
        df_main[name] = vals
        df_main[name] = df_main[name].ffill()
        
    df_main.dropna(inplace=True)
    return df_main

# ローソク足パターン検出
def detect_candle_patterns(df):
    op, hi, lo, cl = df['Open'], df['High'], df['Low'], df['Close']
    
    # 買い
    df['Hammer'] = ta.CDLHAMMER(op, hi, lo, cl)
    df['MorningStar'] = ta.CDLMORNINGSTAR(op, hi, lo, cl)
    df['Piercing'] = ta.CDLPIERCING(op, hi, lo, cl)
    df['ThreeSoldiers'] = ta.CDL3WHITESOLDIERS(op, hi, lo, cl)
    df['Dragonfly'] = ta.CDLDRAGONFLYDOJI(op, hi, lo, cl)

    # 売り
    df['ShootingStar'] = ta.CDLSHOOTINGSTAR(op, hi, lo, cl)
    df['EveningStar'] = ta.CDLEVENINGSTAR(op, hi, lo, cl)
    df['DarkCloud'] = ta.CDLDARKCLOUDCOVER(op, hi, lo, cl)
    df['HangingMan'] = ta.CDLHANGINGMAN(op, hi, lo, cl)
    df['Gravestone'] = ta.CDLGRAVESTONEDOJI(op, hi, lo, cl)
    
    # 両方
    df['Engulfing'] = ta.CDLENGULFING(op, hi, lo, cl)
    return df

# 指標計算
def add_indicators(df):
    data = df.copy()
    c = data['Close'].values; h = data['High'].values; l = data['Low'].values
    
    data['RSI'] = ta.RSI(c, timeperiod=14)
    data['MACD'], _, _ = ta.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    data['BB_UPPER'], _, data['BB_LOWER'] = ta.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2)
    data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
    data['ADX'] = ta.ADX(h, l, c, timeperiod=14)
    
    # VWAP
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    roll_pv = (tp * data['Volume']).rolling(window=20).sum()
    roll_vol = data['Volume'].rolling(window=20).sum()
    data['VWAP_20'] = roll_pv / roll_vol
    data['VWAP_Dev'] = (data['Close'] - data['VWAP_20']) / data['VWAP_20']
    
    # Kalman
    data['Kalman'] = apply_kalman_filter(c)
    
    # パターン
    data = detect_candle_patterns(data)
    
    data.dropna(inplace=True)
    return data

# サポレジ & 形状認識
def calc_sr_and_shapes(df):
    recent = df.tail(500)
    n = 5
    peaks_idx = argrelextrema(recent['High'].values, np.greater, order=n)[0]
    valleys_idx = argrelextrema(recent['Low'].values, np.less, order=n)[0]
    
    candidates = np.concatenate([recent['High'].iloc[peaks_idx].values, recent['Low'].iloc[valleys_idx].values])
    candidates.sort()
    
    levels = []
    current = []
    for p in candidates:
        if not current: current.append(p); continue
        avg = np.mean(current)
        if abs(p - avg)/avg <= 0.02: current.append(p)
        else:
            if len(current) >= 3: levels.append(np.mean(current))
            current = [p]
    if len(current) >= 3: levels.append(np.mean(current))
    
    shapes = []
    # W底
    if len(valleys_idx) >= 2:
        last_v = valleys_idx[-1]; prev_v = valleys_idx[-2]
        p_last = recent['Low'].iloc[last_v]; p_prev = recent['Low'].iloc[prev_v]
        if abs(p_last - p_prev) / p_prev <= 0.03:
            if len(recent) - last_v < 20:
                shapes.append({'Type': 'Double Bottom', 'Signal': 'BUY', 'Price': p_last, 'Date': recent.index[last_v]})
    # M天井
    if len(peaks_idx) >= 2:
        last_p = peaks_idx[-1]; prev_p = peaks_idx[-2]
        p_last = recent['High'].iloc[last_p]; p_prev = recent['High'].iloc[prev_p]
        if abs(p_last - p_prev) / p_prev <= 0.03:
            if len(recent) - last_p < 20:
                shapes.append({'Type': 'Double Top', 'Signal': 'SELL', 'Price': p_last, 'Date': recent.index[last_p]})
    return levels, shapes

# フィボナッチ
def calc_fib(df):
    recent = df.tail(120)
    max_p = recent['High'].max(); min_p = recent['Low'].min()
    diff = max_p - min_p
    return {
        '0.0%': max_p, '23.6%': max_p-diff*0.236, '38.2%': max_p-diff*0.382,
        '50.0%': max_p-diff*0.5, '61.8%': max_p-diff*0.618, '100%': min_p
    }

# 前処理
def preprocess(data, days, feature_cols):
    dataset = data[feature_cols].values
    target = data['Close'].values.reshape(-1, 1)
    scaler_f = MinMaxScaler((0, 1)); scaled_data = scaler_f.fit_transform(dataset)
    scaler_t = MinMaxScaler((0, 1)); scaled_target = scaler_t.fit_transform(target)
    x, y = [], []
    for i in range(days, len(scaled_data)):
        x.append(scaled_data[i-days:i]); y.append(scaled_target[i, 0])
    return np.array(x), np.array(y), scaler_t, scaler_f

# モデル
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1)) 
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# ==============================================================================
# 2. アプリ画面
# ==============================================================================
st.sidebar.header("分析設定")
input_ticker = st.sidebar.text_input("銘柄コード", "2670.T")
prediction_days = st.sidebar.slider("予測期間 (日)", 60, 360, 240)
epochs = st.sidebar.slider("学習回数", 10, 100, 30)

if st.sidebar.button("分析実行"):
    with st.spinner('AIがフル分析中です... (パターン認識・水平線・バックテスト計算)'):
        
        # 1. データ
        try:
            raw_df = fetch_data(input_ticker, '2015-01-01', '2025-12-31')
            df = add_indicators(raw_df)
            sr_levels, chart_shapes = calc_sr_and_shapes(raw_df)
            fib_levels = calc_fib(raw_df)
        except Exception as e:
            st.error(f"エラー: {e}"); st.stop()
        
        feature_cols = ['Close', 'RSI', 'MACD', 'BB_UPPER', 'BB_LOWER', 'Log_Ret', 'VWAP_Dev', 'S&P500', 'USD_JPY', 'VIX']
        
        split = int(len(df) * 0.9)
        train_df = df.iloc[:split]
        test_df = df.iloc[split - prediction_days:]
        
        # 2. 学習
        x_train, y_train, scaler_t, scaler_f = preprocess(train_df, prediction_days, feature_cols)
        model = build_lstm_model((x_train.shape[1], x_train.shape[2]))
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(x_train, y_train, epochs=epochs, batch_size=32, callbacks=[early_stop], verbose=0)
        
        # 3. 予測 & バックテスト (復活！)
        x_test_full, _, _, _ = preprocess(test_df, prediction_days, feature_cols)
        preds_full = scaler_t.inverse_transform(model.predict(x_test_full, verbose=0))
        actual_full = test_df['Close'].values[prediction_days:]
        
        # シャープレシオ計算
        returns = []
        for i in range(len(preds_full) - 1):
            curr = actual_full[i]
            next_pred = preds_full[i+1][0]
            # 上昇予測なら買い、そうでなければノーポジ
            ret = (actual_full[i+1] - curr) / curr if next_pred > curr else 0
            returns.append(ret)
        
        returns = np.array(returns)
        sharpe = 0
        if np.std(returns) != 0:
            sharpe = (np.mean(returns) * 250) / (np.std(returns) * np.sqrt(250))
            
        # 明日の予測
        last_seq = scaler_f.transform(test_df[feature_cols].tail(prediction_days).values)
        pred_price = scaler_t.inverse_transform(model.predict(np.array([last_seq]), verbose=0))[0][0]
        
        # 4. 指標取得
        current_price = df['Close'].iloc[-1]
        vwap = df['VWAP_20'].iloc[-1]
        kalman = df['Kalman'].iloc[-1]
        adx = df['ADX'].iloc[-1]
        atr_val = ta.ATR(raw_df['High'], raw_df['Low'], raw_df['Close'], timeperiod=14).iloc[-1]
        
        # --- 結果表示 ---
        diff = pred_price - current_price
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("現在値", f"{current_price:,.0f} 円")
        c2.metric("AI予測", f"{pred_price:,.0f} 円", f"{diff/current_price*100:+.2f}%")
        
        # AI評価表示
        grade = "C級 (注意)"
        if sharpe > 2.0: grade = "S級 (最強)"
        elif sharpe > 1.0: grade = "A級 (優秀)"
        elif sharpe > 0: grade = "B級 (普通)"
        
        c3.metric("AIモデル評価 (Sharpe)", f"{sharpe:.2f}", grade)
        c4.metric("ADX (トレンド)", f"{adx:.1f}", "強い" if adx>25 else "弱い")
        
        # --- チャート描画 (パターン認識付き) ---
        st.subheader("分析チャート (AI予測 + パターン + 水平線)")
        
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(actual_full, label='Actual', color='gray', alpha=0.6)
        ax.plot(preds_full, label='AI Prediction', color='#00CC66', linewidth=2)
        
        ymin, ymax = min(actual_full), max(actual_full)
        for level in sr_levels:
            if ymin < level < ymax:
                c = 'red' if level > current_price else 'green'
                ax.axhline(level, color=c, linestyle='--', alpha=0.5)
        for name, level in fib_levels.items():
            if ymin < level < ymax:
                ax.axhline(level, color='blue', linestyle=':', alpha=0.3)
        
        # パターンプロット
        chart_start_date = test_df.index[prediction_days]
        full_dates = test_df.index[prediction_days:]
        recent_df = df[df.index >= chart_start_date]
        
        bullish_mask = (recent_df['Hammer']==100) | (recent_df['Engulfing']==100) | \
                       (recent_df['MorningStar']==100) | (recent_df['Piercing']==100) | \
                       (recent_df['ThreeSoldiers']==100) | (recent_df['Dragonfly']==100)
                       
        bearish_mask = (recent_df['ShootingStar']==-100) | (recent_df['Engulfing']==-100) | \
                       (recent_df['EveningStar']==-100) | (recent_df['DarkCloud']==-100) | \
                       (recent_df['HangingMan']==-100) | (recent_df['Gravestone']==-100)

        for date_idx in recent_df[bullish_mask].index:
            if date_idx in full_dates:
                pos = full_dates.get_loc(date_idx)
                ax.scatter(pos, recent_df.loc[date_idx, 'Low']*0.99, marker='^', color='red', s=80, zorder=5)

        for date_idx in recent_df[bearish_mask].index:
            if date_idx in full_dates:
                pos = full_dates.get_loc(date_idx)
                ax.scatter(pos, recent_df.loc[date_idx, 'High']*1.01, marker='v', color='blue', s=80, zorder=5)
                
        for s in chart_shapes:
            if s['Date'] in full_dates:
                pos = full_dates.get_loc(s['Date'])
                marker = 'W' if s['Signal']=='BUY' else 'M'
                color = 'magenta' if s['Signal']=='BUY' else 'cyan'
                ax.scatter(pos, s['Price'], marker=f'${marker}$', s=150, color=color, zorder=10)

        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # --- 詳細レポート ---
        st.divider()
        st.subheader("📊 最終トレードプラン & 詳細分析")
        
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown("#### ① 直近のパターン分析")
            last_candle = df.iloc[-1]
            found = False
            if last_candle['MorningStar'] == 100: st.success("🔥 **[今日] 明けの明星**: 最強クラスの買い転換サイン"); found=True
            if last_candle['ThreeSoldiers'] == 100: st.success("🔥 **[今日] 赤三兵**: 上昇トレンド決定打"); found=True
            if last_candle['Piercing'] == 100: st.success("🔥 **[今日] 切り込み線**: 強い反発サイン"); found=True
            if last_candle['Hammer'] == 100: st.success("🔥 **[今日] ハンマー/たくり線**: 底打ちサイン"); found=True
            if last_candle['Dragonfly'] == 100: st.success("🔥 **[今日] トンボ**: 強力な買い支え"); found=True
            if last_candle['Engulfing'] == 100: st.success("🔥 **[今日] 強気包み足**: 強い買いサイン"); found=True
            
            if last_candle['EveningStar'] == -100: st.error("💧 **[今日] 宵の明星**: 最強クラスの売り転換サイン"); found=True
            if last_candle['DarkCloud'] == -100: st.error("💧 **[今日] かぶせ線**: 失速のサイン"); found=True
            if last_candle['HangingMan'] == -100: st.error("💧 **[今日] 首吊り線**: 天井警戒サイン"); found=True
            if last_candle['ShootingStar'] == -100: st.error("💧 **[今日] 流れ星**: 上ヒゲ天井"); found=True
            if last_candle['Engulfing'] == -100: st.error("💧 **[今日] 弱気包み足**: 売り転換"); found=True

            for s in chart_shapes:
                if (df.index[-1] - s['Date']).days < 20:
                    st.info(f"⚡ **{s['Type']}** 検知 ({s['Date'].date()}): {s['Signal']}サイン")
                    found = True
            
            if not found: st.write("直近に特異なローソク足パターンは見当たりません。")
            
            st.markdown("#### ② 環境認識")
            if current_price < vwap: st.success(f"✅ **VWAP割安**: 機関の買いゾーン ({vwap:.0f}円より下)")
            else: st.warning(f"⚠ **VWAP割高**: {vwap:.0f}円より上")
            
            if current_price < kalman: st.caption(f"Kalman(真の値): {kalman:.0f}円 (割安圏)")
            else: st.caption(f"Kalman(真の値): {kalman:.0f}円 (加熱圏)")

        with rc2:
            st.markdown("#### ③ AI推奨アクション")
            
            if pred_price > current_price:
                st.success(f"### 判定: 【買い (BUY)】")
                entry = current_price - (atr_val * 0.2)
                if current_price > vwap and (current_price - vwap) < atr_val: entry = vwap
                stop = entry - (atr_val * 1.5)
                
                nearest_sup = [l for l in sr_levels if l < current_price]
                if nearest_sup and (entry - nearest_sup[-1]) < atr_val * 1.5:
                    stop = nearest_sup[-1] - (atr_val * 0.5)
                    st.caption("※損切りを支持線の下に調整しました")
                
                target = max(pred_price, entry + (atr_val * 2))
                rr = (target - entry) / (entry - stop)
                
                c_a, c_b, c_c = st.columns(3)
                c_a.metric("エントリー指値", f"{entry:,.0f} 円")
                c_b.metric("損切り", f"{stop:,.0f} 円")
                c_c.metric("利確目標", f"{target:,.0f} 円")
                
                if rr > 1.5: st.caption(f"★ 期待値(R/R): {rr:.2f} (合格)")
                else: st.caption(f"⚠ 期待値(R/R): {rr:.2f} (リスク高め)")
                
            elif pred_price < current_price:
                st.error(f"### 判定: 【売り (SELL)】")
                entry = current_price + (atr_val * 0.2)
                stop = entry + (atr_val * 1.5)
                target = min(pred_price, entry - (atr_val * 2))
                
                c_a, c_b, c_c = st.columns(3)
                c_a.metric("エントリー指値", f"{entry:,.0f} 円")
                c_b.metric("損切り", f"{stop:,.0f} 円")
                c_c.metric("利確目標", f"{target:,.0f} 円")
            else:
                st.write("様子見")
            
            st.markdown("---")
            st.markdown("#### ④ 水平線アラート")
            nearest_res = [l for l in sr_levels if l > current_price]
            nearest_sup = [l for l in sr_levels if l < current_price]
            
            if nearest_res:
                dist = nearest_res[0] - current_price
                if dist < atr_val: st.error(f"⚠ **壁接近**: すぐ上 {nearest_res[0]:.0f}円 (あと{dist:.0f}円)")
                else: st.write(f"上の抵抗線: {nearest_res[0]:.0f}円")
            else: st.write("上の抵抗線: なし")
            
            if nearest_sup:
                dist = current_price - nearest_sup[-1]
                if dist < atr_val: st.success(f"🛡️ **床あり**: すぐ下 {nearest_sup[-1]:.0f}円 (下{dist:.0f}円)")
                else: st.write(f"下の支持線: {nearest_sup[-1]:.0f}円")