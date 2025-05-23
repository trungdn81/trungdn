import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from vnstock import Vnstock
from datetime import datetime, timedelta
import time
import io

def update_price_cache(symbol, start_date, end_date, source="VCI"):
    try:
        path = f"cache/{symbol}.csv"
        if os.path.exists(path):
            old_df = pd.read_csv(path)
            last_date = pd.to_datetime(old_df['time']).max()
            fetch_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            old_df = pd.DataFrame()
            fetch_start = start_date
        if fetch_start > end_date:
            print(f"✅ {symbol} đã đầy đủ dữ liệu.")
            return
        new_df = stock_historical_data(symbol, fetch_start, end_date, resolution="1D", type=source.lower())
        full_df = pd.concat([old_df, new_df]).drop_duplicates(subset=["time"]).sort_values("time")
        full_df.to_csv(path, index=False)
        print(f"✅ Cập nhật dữ liệu giá cho {symbol} thành công.")
    except Exception as e:
        print(f"❌ Lỗi khi cập nhật giá cho {symbol}: {e}")

# ==== Thông tin logic chỉ báo ==== #
logic_info = {
    "MA20": {"mota": "Giá hiện tại cắt lên MA20", "vaitro": "Xác định xu hướng ngắn hạn", "uutien": "⭐⭐⭐"},
    "MACD": {"mota": "MACD > Signal và cắt lên", "vaitro": "Tín hiệu đảo chiều", "uutien": "⭐⭐⭐⭐"},
    "RSI": {"mota": "RSI từ 50–80", "vaitro": "Đo sức mạnh giá", "uutien": "⭐⭐"},
    "BB": {"mota": "Giá vượt dải BB trên", "vaitro": "Breakout", "uutien": "⭐"},
    "ADX": {"mota": "ADX > 20", "vaitro": "Xác nhận xu hướng mạnh", "uutien": "⭐⭐"},
}

# ==== Các hàm xử lý kỹ thuật ==== #
def compute_indicators(df):
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
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_adx(df, n=14):
    high, low, close = df['high'], df['low'], df['close']
    plus_dm = high.diff().where((high.diff() > low.diff()) & (high.diff() > 0), 0.0)
    minus_dm = low.diff().where((low.diff() > high.diff()) & (low.diff() > 0), 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    plus_di = 100 * (plus_dm.rolling(n).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(n).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(n).mean()

def score_stock(df, selected_indicators, weights):
    score, reasons, logic = 0, [], {
        "MA20": df['close'].iloc[-1] > df['MA20'].iloc[-1] and df['close'].iloc[-2] < df['MA20'].iloc[-2],
        "MACD": df['MACD'].iloc[-1] > df['Signal'].iloc[-1] and df['MACD'].iloc[-2] < df['Signal'].iloc[-2],
        "RSI": 50 <= df['RSI'].iloc[-1] <= 80,
        "BB": df['close'].iloc[-1] > df['UpperBB'].iloc[-1],
        "ADX": df['ADX'].iloc[-1] > 20
    }
    for k, v in logic.items():
        if not selected_indicators.get(k, True):
            continue
        if v:
            w = weights.get(k, 1)
            score += w
            reasons.append(f"{k}(+{w})")
    return score, ", ".join(reasons), logic

st.sidebar.markdown("### ℹ️ Thông tin chỉ báo")
for key, val in logic_info.items():
    st.sidebar.markdown(f"**{key}**: {val['mota']} – {val['vaitro']} ({val['uutien']})")

# ==== Preset chiến lược lọc nâng cao (Khôi phục tự động) ==== #
st.sidebar.markdown("### 🎯 Preset chiến lược lọc")

presets = {
    "Mặc định": {"selected_indicators": {k: True for k in logic_info}, "weights": {k: 1 for k in logic_info}},
    "Breakout": {"selected_indicators": {"MACD": True, "BB": True, "RSI": False, "MA20": False, "ADX": False}, "weights": {"MACD": 2, "BB": 2, "RSI": 0, "MA20": 0, "ADX": 0}},
    "Trend-following": {"selected_indicators": {"MACD": True, "MA20": True, "ADX": True, "RSI": False, "BB": False}, "weights": {"MACD": 2, "MA20": 2, "ADX": 2, "RSI": 0, "BB": 0}},
    "Rebound": {"selected_indicators": {"MACD": True, "RSI": True, "MA20": True, "BB": False, "ADX": False}, "weights": {"MACD": 2, "RSI": 2, "MA20": 1, "BB": 0, "ADX": 0}},
    "Volume spike": {"selected_indicators": {"MACD": True, "MA20": True, "RSI": True, "BB": False, "ADX": True}, "weights": {"MACD": 1, "MA20": 2, "RSI": 1, "BB": 0, "ADX": 2}},
    "Breakout RSI": {"selected_indicators": {"MACD": True, "RSI": True, "BB": True, "MA20": False, "ADX": False}, "weights": {"MACD": 2, "RSI": 2, "BB": 2, "MA20": 0, "ADX": 0}},
    "Vượt đỉnh 52 tuần": {"selected_indicators": {"MACD": True, "MA20": True, "ADX": True, "RSI": False, "BB": False}, "weights": {"MACD": 2, "MA20": 2, "ADX": 2, "RSI": 0, "BB": 0}},
    "EMA crossover": {"selected_indicators": {"MACD": True, "MA20": False, "ADX": False, "RSI": False, "BB": False}, "weights": {"MACD": 3, "MA20": 0, "ADX": 0, "RSI": 0, "BB": 0}}
}

preset_descriptions = {
    "Mặc định": "🔹 Bật tất cả chỉ báo, trọng số = 1",
    "Breakout": "🔹 Ưu tiên BB, MACD – Dùng cho cổ phiếu breakout",
    "Trend-following": "🔹 Ưu tiên MA20, ADX, MACD – Theo xu hướng",
    "Rebound": "🔹 MACD, RSI, MA20 – Hồi phục từ vùng quá bán",
    "Volume spike": "🔹 MA20, ADX, RSI, MACD – Bùng nổ thanh khoản",
    "Breakout RSI": "🔹 RSI > 70 + MACD cắt lên – Breakout",
    "Vượt đỉnh 52 tuần": "🔹 Giá vượt đỉnh cũ 52W",
    "EMA crossover": "🔹 EMA12 cắt EMA26 + MACD – Đảo chiều"
}

preset_option = st.sidebar.selectbox("Chọn preset", list(presets.keys()))
st.sidebar.markdown(f"ℹ️ **Chiến lược '{preset_option}':** {preset_descriptions[preset_option]}")

for key in logic_info:
    sel_key = f"selected_{key}"
    weight_key = f"weight_{key}"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = presets["Mặc định"]["selected_indicators"][key]
    if weight_key not in st.session_state:
        st.session_state[weight_key] = presets["Mặc định"]["weights"][key]

if st.sidebar.button("🔄 Áp dụng preset"):
    for key in logic_info:
        st.session_state[f"selected_{key}"] = presets[preset_option]["selected_indicators"][key]
        st.session_state[f"weight_{key}"] = presets[preset_option]["weights"][key]
    st.sidebar.success("✅ Đã khôi phục chỉ số và trọng số theo preset!")

st.sidebar.markdown("### 🧪 Tuỳ chỉnh chỉ báo & trọng số")
selected_indicators = {}
weights = {}
for key in logic_info:
    col1, col2 = st.sidebar.columns([2, 1])
    selected_indicators[key] = col1.checkbox(f"Bật {key}", value=st.session_state[f"selected_{key}"], key=f"selected_{key}")
    weights[key] = col2.number_input(f"W{key}", min_value=0, max_value=5, value=st.session_state[f"weight_{key}"], step=1, key=f"weight_{key}")

# ==== Preset nâng cao với điều kiện lọc ==== #
extra_conditions = {
    "Mặc định": {},
    "Breakout": {"price_change_5d_min": 3, "volume_ratio_min": 1.5},
    "Trend-following": {"price_min": 20000, "price_max": 80000, "ma50_break": True},
    "Rebound": {"price_change_5d_max": 0, "price_change_5d_min": -5},
    "Volume spike": {"volume_ratio_min": 2, "price_min": 10000, "price_max": 50000},
    "Breakout RSI": {"rsi_min": 70, "price_change_5d_min": 2},
    "Vượt đỉnh 52 tuần": {"ma100_break": True},
    "EMA crossover": {"macd_positive": True}
}

# Hàm an toàn ép kiểu
def safe_int(val, default):
    try:
        return int(val)
    except:
        return default

def safe_float(val, default):
    try:
        return float(val)
    except:
        return default

# UI sidebar
st.sidebar.markdown("### ⚙️ Điều kiện lọc nâng cao")
selected_preset = st.sidebar.selectbox("Chọn preset chiến lược", list(extra_conditions.keys()))
if st.sidebar.button("🔄 Áp dụng điều kiện lọc"):
    st.session_state['preset_conditions'] = extra_conditions[selected_preset]

# Lấy preset hiện tại từ session state
preset_condition = st.session_state.get('preset_conditions', extra_conditions[selected_preset])

# Các điều kiện lọc - Đảm bảo ép kiểu đúng
price_min = st.sidebar.number_input(
    "Giá tối thiểu (VND)", min_value=0, max_value=1_000_000,
    value=safe_int(preset_condition.get("price_min"), 0), step=1000
)

price_max = st.sidebar.number_input(
    "Giá tối đa (VND)", min_value=0, max_value=1_000_000,
    value=safe_int(preset_condition.get("price_max"), 200000), step=1000
)

price_change_5d_min = st.sidebar.slider(
    "Tăng giá 5 phiên gần nhất tối thiểu (%)", -20, 20,
    value=safe_int(preset_condition.get("price_change_5d_min"), -20)
)

price_change_5d_max = st.sidebar.slider(
    "Tăng giá 5 phiên gần nhất tối đa (%)", -20, 20,
    value=safe_int(preset_condition.get("price_change_5d_max"), 20)
)

volume_ratio_min = st.sidebar.slider(
    "Volume hôm nay lớn hơn bao nhiêu lần TB20", 0.0, 5.0,
    step=0.1, value=safe_float(preset_condition.get("volume_ratio_min"), 0.0)
)

rsi_min = st.sidebar.slider(
    "RSI tối thiểu", 0, 100,
    value=safe_int(preset_condition.get("rsi_min"), 0)
)
# Tổng hợp điều kiện lọc thành dict để truyền về sau
ma50_break = st.sidebar.checkbox("Giá vượt MA50", value=bool(preset_condition.get("ma50_break", False)))
ma100_break = st.sidebar.checkbox("Giá vượt MA100", value=bool(preset_condition.get("ma100_break", False)))
macd_positive = st.sidebar.checkbox("MACD dương", value=bool(preset_condition.get("macd_positive", False)))

extra_filters = {
    "price_min": price_min,
    "price_max": price_max,
    "price_change_5d_min": price_change_5d_min,
    "price_change_5d_max": price_change_5d_max,
    "volume_ratio_min": volume_ratio_min,
    "ma50_break": ma50_break,
    "ma100_break": ma100_break,
    "macd_positive": macd_positive,
    "rsi_min": rsi_min
}

# st.sidebar.success("🎯 Điều kiện lọc nâng cao đã được áp dụng theo preset!")

# ==== Phần lọc dữ liệu và xuất Excel ==== #
st.title("📊 Bộ lọc kỹ thuật cổ phiếu nâng cao")

step = st.radio("Chọn thao tác:", ["Bước 1: Cập nhật dữ liệu cache", "Bước 2: Lọc kỹ thuật"])
start_date = st.date_input("Ngày bắt đầu", datetime.now() - timedelta(days=90))
end_date = st.date_input("Ngày kết thúc", datetime.now())
data_source = st.selectbox("Nguồn dữ liệu", ["VCI", "TCBS"])

if step == "Bước 1: Cập nhật dữ liệu cache":
    uploaded = st.file_uploader("📥 Tải file CSV danh sách mã (cột 'symbol', 'exchange')", type=["csv"])
    if uploaded:
        df_input = pd.read_csv(uploaded)
        df_input.columns = [c.strip().lower() for c in df_input.columns]
        if 'symbol' not in df_input or 'exchange' not in df_input:
            st.error("❌ File phải có cột 'symbol' và 'exchange'")
            st.stop()
        sàn_chọn = st.multiselect("Chọn sàn cần tải cache", df_input['exchange'].unique().tolist(), default=df_input['exchange'].unique().tolist())
        symbols = df_input[df_input['exchange'].isin(sàn_chọn)]['symbol'].dropna().unique().tolist()
        if st.button("🚀 Cập nhật cache"):
            for i, symbol in enumerate(symbols):
                st.write(f"📈 {symbol} ({i+1}/{len(symbols)})")
                update_price_cache(symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), source=data_source)
            st.success("✅ Đã cập nhật xong!")

elif step == "Bước 2: Lọc kỹ thuật":
    min_volume = st.number_input("Volume TB tối thiểu (20 phiên)", value=100000, step=50000)
    min_score = st.slider("Điểm lọc tối thiểu", 1, 10, 3)

    if st.button("🚀 Bắt đầu lọc kỹ thuật"):
        # st.stop()

        result, logic_counts = [], {k: 0 for k in logic_info}
        cache_dir = "cache"
        if os.path.exists(cache_dir):
            files = [f for f in os.listdir(cache_dir) if f.endswith(".csv")]
            for file in files:
                symbol = file.replace(".csv", "")
                try:
                    df = pd.read_csv(os.path.join(cache_dir, file))
                    if len(df) < 30: continue
                    df = compute_indicators(df)

                    price = df['close'].iloc[-1]
                    change_5d = (df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6] * 100
                    volume_today = df['volume'].iloc[-1]
                    volume_avg = df['volume'].rolling(20).mean().iloc[-1]
                    volume_ratio = volume_today / volume_avg if volume_avg > 0 else 0
                    ma50 = df['close'].iloc[-1] > df['close'].rolling(50).mean().iloc[-1]
                    ma100 = df['close'].iloc[-1] > df['close'].rolling(100).mean().iloc[-1]
                    macd_pos = df['MACD'].iloc[-1] > 0
                    rsi = df['RSI'].iloc[-1]

                    if price < extra_filters['price_min'] or price > extra_filters['price_max']:
                        continue
                    if change_5d < extra_filters['price_change_5d_min'] or change_5d > extra_filters['price_change_5d_max']:
                        continue
                    if volume_ratio < extra_filters['volume_ratio_min']:
                        continue
                    if extra_filters['ma50_break'] and not ma50:
                        continue
                    if extra_filters['ma100_break'] and not ma100:
                        continue
                    if extra_filters['macd_positive'] and not macd_pos:
                        continue
                    if rsi < extra_filters['rsi_min']:
                        continue

                    score, note, logic = score_stock(df, {k: selected_indicators[k] for k in logic_info}, weights)
                    for k, v in logic.items():
                        if v: logic_counts[k] += 1
                    if score >= min_score:
                        result.append({"Mã": symbol, "Điểm": score, "Chi tiết": note})
                except Exception as e:
                    st.warning(f"❌ {symbol}: lỗi {e}")

            if result:
                df_result = pd.DataFrame(result).sort_values("Điểm", ascending=False)
                st.success(f"✅ Có {len(df_result)} mã đạt điểm ≥ {min_score}")
                st.dataframe(df_result)

                logic_df = pd.DataFrame.from_dict(logic_counts, orient='index', columns=['Số mã đạt'])
                logic_df['%'] = (logic_df['Số mã đạt'] / len(files) * 100).round(1)
                st.markdown("### 📊 Thống kê chỉ báo kỹ thuật")
                st.dataframe(logic_df)

                # ✅ Chỉ tạo file Excel khi có ít nhất một sheet có dữ liệu
                if not df_result.empty or not logic_df.empty:
                    with io.BytesIO() as output:
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            if not df_result.empty:
                                df_result.to_excel(writer, sheet_name="Ket_qua", index=False)
                            if not logic_df.empty:
                                logic_df.to_excel(writer, sheet_name="Thong_ke", index=True)
                        st.download_button(
                            label="📥 Lưu kết quả ra Excel",
                            data=output.getvalue(),
                            file_name="ket_qua_loc_ky_thuat.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            else:
                st.warning("❗ Không có mã nào đạt đủ điều kiện.")
