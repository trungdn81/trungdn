# loc_ky_thuat_streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
from vnstock import Vnstock
from datetime import datetime, timedelta
from vnstock import listing_companies
import io

st.set_page_config(page_title="Lọc cổ phiếu kỹ thuật toàn thị trường", layout="wide")

# ======================== CÀI ĐẶT =============================
start_date = st.date_input("Chọn ngày bắt đầu", datetime.now() - timedelta(days=120))
end_date = st.date_input("Chọn ngày kết thúc", datetime.now())

resolution = "1D"

st.markdown("""
## 🧠 Nguyên lý lọc kỹ thuật:
| Chỉ báo | Ý nghĩa | Điều kiện lọc |
|--------|--------|-----------------------------|
| MA20   | Xu hướng trung hạn | Close > MA20 và cắt lên |
| MACD   | Động lượng | MACD > Signal và cắt lên |
| RSI    | Sức mạnh xu hướng | RSI trong khoảng 50–80 |
| BB     | Biến động | Giá vượt band trên BB |
| ADX    | Độ mạnh xu hướng | ADX > 20 |
""")

# ======================== CHỈ BÁO =============================
def add_indicators(df):
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['RSI'] = compute_rsi(df['close'])
    df['20SD'] = df['close'].rolling(window=20).std()
    df['UpperBB'] = df['MA20'] + 2 * df['20SD']
    df['LowerBB'] = df['MA20'] - 2 * df['20SD']
    df['ADX'] = compute_adx(df)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_adx(df, n=14):
    high = df['high']
    low = df['low']
    close = df['close']
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    plus_di = 100 * (plus_dm.rolling(n).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(n).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(n).mean()

# ======================== LOGIC LỌC =============================
def score_stock(df):
    score = 0
    reasons = []
    if df['close'].iloc[-1] > df['MA20'].iloc[-1] and df['close'].iloc[-2] < df['MA20'].iloc[-2]:
        score += 1
        reasons.append("MA20: cắt lên và trên MA20")
    if df['MACD'].iloc[-1] > df['Signal'].iloc[-1] and df['MACD'].iloc[-2] < df['Signal'].iloc[-2]:
        score += 1
        reasons.append("MACD: cắt lên signal")
    if 50 <= df['RSI'].iloc[-1] <= 80:
        score += 1
        reasons.append("RSI: trong vùng mạnh (50–80)")
    if df['close'].iloc[-1] > df['UpperBB'].iloc[-1]:
        score += 1
        reasons.append("BB: giá vượt dải trên")
    if df['ADX'].iloc[-1] > 20:
        score += 1
        reasons.append("ADX: xu hướng mạnh")
    return score, ", ".join(reasons)

# ======================== THỰC THI =============================
if st.button("🚀 Bắt đầu lọc cổ phiếu kỹ thuật"):
    all_symbols = listing_companies()['ticker'].tolist()
    result = []
    progress = st.progress(0)

    for i, symbol in enumerate(all_symbols):
        try:
            df = Vnstock().stock(symbol=symbol, source="TCBS").price.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                resolution=resolution
            )
            if df is None or len(df) < 30:
                continue
            df = add_indicators(df)
            score, note = score_stock(df)
            if score >= 3:
                result.append({"Mã": symbol, "Điểm": score, "Chi tiết": note})
        except:
            continue
        progress.progress((i + 1) / len(all_symbols))

    if result:
        df_result = pd.DataFrame(result).sort_values(by="Điểm", ascending=False)
        st.success(f"✅ Có {len(df_result)} mã cổ phiếu đạt điều kiện!")
        st.dataframe(df_result)

        with io.BytesIO() as output:
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_result.to_excel(writer, index=False)
            st.download_button(
                label="📥 Tải kết quả Excel",
                data=output.getvalue(),
                file_name="ket_qua_loc_ky_thuat.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.warning("Không có mã nào đạt đủ điều kiện.")
