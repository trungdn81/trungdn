# === PHẦN 1: Import thư viện & tải cache từ Google Drive ===
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import shutil
import gdown
from vnstock import Vnstock
from datetime import datetime, timedelta
import time
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Tạo thư mục cache nếu chưa có
if not os.path.exists("cache"):
    os.makedirs("cache")

# Hàm tải file zip từ Google Drive và giải nén vào thư mục cache
def download_and_extract_cache(drive_file_id):
    url = f'https://drive.google.com/uc?id={drive_file_id}'
    output = 'cache.zip'
    gdown.download(url, output, quiet=False)
    shutil.unpack_archive(output, 'cache', 'zip')

# Giao diện tải dữ liệu từ Google Drive
st.sidebar.markdown("### 📂 Dữ liệu cache từ Google Drive")
drive_cache_id = st.sidebar.text_input(
    "📥 Nhập Google Drive File ID (.zip chứa cache)",
    value="1AMrekvoERyho9vpudK-U9UkK712FoGbt",
    help="Ví dụ: ID từ https://drive.google.com/file/d/1abcXYZ/view => ID là phần giữa"
)
if st.sidebar.button("📦 Tải và giải nén cache"):
    try:
        download_and_extract_cache(drive_cache_id)
        st.sidebar.success("✅ Đã tải cache từ Google Drive thành công!")
    except Exception as e:
        st.sidebar.error(f"❌ Lỗi khi tải cache: {e}")
# === PHẦN 2: Logic chỉ báo & preset chiến lược ===

# ==== Thông tin logic chỉ báo ==== #
logic_info = {
    "MA20": {"mota": "Giá hiện tại cắt lên MA20", "vaitro": "Xác định xu hướng ngắn hạn", "uutien": "⭐⭐⭐"},
    "MACD": {"mota": "MACD > Signal và cắt lên", "vaitro": "Tín hiệu đảo chiều", "uutien": "⭐⭐⭐⭐"},
    "RSI": {"mota": "RSI từ 50–80", "vaitro": "Đo sức mạnh giá", "uutien": "⭐⭐"},
    "BB": {"mota": "Giá vượt dải BB trên", "vaitro": "Breakout", "uutien": "⭐"},
    "ADX": {"mota": "ADX > 20", "vaitro": "Xác nhận xu hướng mạnh", "uutien": "⭐⭐"},
}

# Sidebar: Thông tin chỉ báo
st.sidebar.markdown("### ℹ️ Thông tin chỉ báo")
for key, val in logic_info.items():
    st.sidebar.markdown(f"**{key}**: {val['mota']} – {val['vaitro']} ({val['uutien']})")

# Preset lọc nâng cao
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

# Khởi tạo session state mặc định nếu chưa có
for key in logic_info:
    sel_key = f"selected_{key}"
    weight_key = f"weight_{key}"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = presets["Mặc định"]["selected_indicators"][key]
    if weight_key not in st.session_state:
        st.session_state[weight_key] = presets["Mặc định"]["weights"][key]

# Nút áp dụng preset
if st.sidebar.button("🔄 Áp dụng preset"):
    for key in logic_info:
        st.session_state[f"selected_{key}"] = presets[preset_option]["selected_indicators"][key]
        st.session_state[f"weight_{key}"] = presets[preset_option]["weights"][key]
    st.sidebar.success("✅ Đã khôi phục chỉ số và trọng số theo preset!")
# === Giao diện điều chỉnh logic lọc & trọng số từng chỉ báo ===
st.sidebar.markdown("### 🧠 Cài đặt lọc kỹ thuật chi tiết")

for key in logic_info:
    cols = st.sidebar.columns([0.4, 0.6])
    st.session_state[f"selected_{key}"] = cols[0].checkbox(
        f"{key}", value=st.session_state.get(f"selected_{key}", True))
    st.session_state[f"weight_{key}"] = cols[1].number_input(
        f"Trọng số {key}", value=st.session_state.get(f"weight_{key}", 1), step=1, min_value=0, key=f"input_weight_{key}")
    
# === PHẦN 3: Giao diện điều kiện lọc nâng cao ===

# ==== Preset điều kiện lọc nâng cao ==== #
st.sidebar.markdown("### ⚙️ Điều kiện lọc nâng cao")

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

# Hàm ép kiểu an toàn
def safe_int(val, default):
    try: return int(val)
    except: return default

def safe_float(val, default):
    try: return float(val)
    except: return default

# Giao diện chọn preset điều kiện lọc
selected_preset = st.sidebar.selectbox("Chọn preset chiến lược", list(extra_conditions.keys()))
if st.sidebar.button("🔄 Áp dụng điều kiện lọc"):
    st.session_state['preset_conditions'] = extra_conditions[selected_preset]

# Lấy điều kiện lọc hiện tại từ session hoặc preset mặc định
preset_condition = st.session_state.get('preset_conditions', extra_conditions[selected_preset])

# Giao diện các điều kiện lọc nâng cao
price_min = st.sidebar.number_input("Giá tối thiểu (VND)", min_value=0, max_value=1_000_000, value=safe_int(preset_condition.get("price_min"), 0), step=1000)
price_max = st.sidebar.number_input("Giá tối đa (VND)", min_value=0, max_value=1_000_000, value=safe_int(preset_condition.get("price_max"), 200000), step=1000)
price_change_5d_min = st.sidebar.slider("Tăng giá 5 phiên gần nhất tối thiểu (%)", -20, 20, value=safe_int(preset_condition.get("price_change_5d_min"), -20))
price_change_5d_max = st.sidebar.slider("Tăng giá 5 phiên gần nhất tối đa (%)", -20, 20, value=safe_int(preset_condition.get("price_change_5d_max"), 20))
volume_ratio_min = st.sidebar.slider("Volume hôm nay lớn hơn bao nhiêu lần TB20", 0.0, 5.0, step=0.1, value=safe_float(preset_condition.get("volume_ratio_min"), 0.0))
rsi_min = st.sidebar.slider("RSI tối thiểu", 0, 100, value=safe_int(preset_condition.get("rsi_min"), 0))

ma50_break = st.sidebar.checkbox("Giá vượt MA50", value=bool(preset_condition.get("ma50_break", False)))
ma100_break = st.sidebar.checkbox("Giá vượt MA100", value=bool(preset_condition.get("ma100_break", False)))
macd_positive = st.sidebar.checkbox("MACD dương", value=bool(preset_condition.get("macd_positive", False)))

# Gom lại thành dict để truyền vào bộ lọc
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

# === PHẦN 4: Giao diện chính lọc kỹ thuật, tính điểm và thống kê ===

st.title("📊 Bộ lọc kỹ thuật cổ phiếu nâng cao")

# Giao diện lọc chính
min_volume = st.number_input("Volume TB tối thiểu (20 phiên)", value=100000, step=50000)
min_score = st.slider("Điểm lọc tối thiểu", 1, 10, 3)

# === Tải file danh sách mã ===
st.markdown("## 📄 Tải danh sách mã để lọc")

uploaded = st.file_uploader("📥 Tải file CSV danh sách mã (gồm cột 'symbol' và 'exchange')", type=["csv"])
if uploaded:
    df_input = pd.read_csv(uploaded)
    df_input.columns = [c.strip().lower() for c in df_input.columns]
    if 'symbol' not in df_input or 'exchange' not in df_input:
        st.error("❌ File phải có cột 'symbol' và 'exchange'")
        st.stop()
    sàn_chọn = st.multiselect("Chọn sàn cần lọc", df_input['exchange'].unique().tolist(), default=df_input['exchange'].unique().tolist())
    symbols = df_input[df_input['exchange'].isin(sàn_chọn)]['symbol'].dropna().unique().tolist()
    st.success(f"✅ Đã chọn {len(symbols)} mã để lọc")
else:
    st.warning("⚠️ Vui lòng upload file danh_sach_ma.csv trước khi lọc")
    symbols = []

if st.button("🚀 Bắt đầu lọc kỹ thuật"):
    result, logic_counts = [], {k: 0 for k in logic_info}
    cache_dir = "cache"
    if os.path.exists(cache_dir):
        files = [f"{symbol}.csv" for symbol in symbols if os.path.exists(f"cache/{symbol}.csv")]
        for file in files:
            symbol = file.replace(".csv", "")
            try:
                df = pd.read_csv(os.path.join(cache_dir, file))
                if 'date' in df.columns and 'time' not in df.columns:
                    df.rename(columns={'date': 'time'}, inplace=True)
                df['time'] = pd.to_datetime(df['time'])

                if len(df) < 30:
                    continue

                # === Tính toán các chỉ báo kỹ thuật ===
                df['MA20'] = df['close'].rolling(window=20).mean()
                df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
                df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = df['EMA12'] - df['EMA26']
                df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = -delta.where(delta < 0, 0).rolling(14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                df['20SD'] = df['close'].rolling(window=20).std()
                df['UpperBB'] = df['MA20'] + 2 * df['20SD']
                df['LowerBB'] = df['MA20'] - 2 * df['20SD']
                high, low, close = df['high'], df['low'], df['close']
                plus_dm = high.diff().where((high.diff() > low.diff()) & (high.diff() > 0), 0.0)
                minus_dm = low.diff().where((low.diff() > high.diff()) & (low.diff() > 0), 0.0)
                tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
                atr = tr.rolling(14).mean()
                plus_di = 100 * (plus_dm.rolling(14).sum() / atr)
                minus_di = 100 * (minus_dm.rolling(14).sum() / atr)
                dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
                df['ADX'] = dx.rolling(14).mean()

                # === Tính các điều kiện lọc nâng cao ===
                price = df['close'].iloc[-1]
                change_5d = (df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6] * 100
                volume_today = df['volume'].iloc[-1]
                volume_avg = df['volume'].rolling(20).mean().iloc[-1]
                volume_ratio = volume_today / volume_avg if volume_avg > 0 else 0
                ma50 = df['close'].iloc[-1] > df['close'].rolling(50).mean().iloc[-1]
                ma100 = df['close'].iloc[-1] > df['close'].rolling(100).mean().iloc[-1]
                macd_pos = df['MACD'].iloc[-1] > 0
                rsi = df['RSI'].iloc[-1]

                # Lọc theo điều kiện nâng cao
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
                if volume_avg < min_volume:
                    continue

                # === Tính điểm kỹ thuật ===
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

                score, note, logic = score_stock(
                    df,
                    {k: st.session_state[f"selected_{k}"] for k in logic_info},
                    {k: st.session_state[f"weight_{k}"] for k in logic_info}
                )

                for k, v in logic.items():
                    if v:
                        logic_counts[k] += 1

                if score >= min_score:
                    result.append({"Mã": symbol, "Điểm": score, "Chi tiết": note})

            except Exception as e:
                st.warning(f"❌ {symbol}: lỗi {e}")

    if result:
        df_result = pd.DataFrame(result).sort_values("Điểm", ascending=False)
        st.success(f"✅ Có {len(df_result)} mã đạt điểm ≥ {min_score}")
        st.dataframe(df_result)

        # === Xuất kết quả ra Excel ===
        import io
        import base64

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_result.to_excel(writer, sheet_name='Top Results', index=False)
            writer.close()
            b64 = base64.b64encode(buffer.getvalue()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="ket_qua_loc_ky_thuat.xlsx">📥 Tải kết quả Excel</a>'
            st.markdown(href, unsafe_allow_html=True)

        # Thống kê số lượng mã đạt theo từng chỉ báo
        logic_df = pd.DataFrame.from_dict(logic_counts, orient='index', columns=['Số mã đạt'])
        logic_df['%'] = (logic_df['Số mã đạt'] / len(files) * 100).round(1)
        st.markdown("### 📊 Thống kê chỉ báo kỹ thuật")
        st.dataframe(logic_df)
    else:
        st.warning("❗ Không có mã nào đạt đủ điều kiện.")

# === PHẦN 5: Biểu đồ kỹ thuật tổng hợp cho từng mã ===

st.markdown("## 📈 Xem biểu đồ kỹ thuật cho mã bất kỳ")

# Lấy danh sách mã từ thư mục cache
available_files = sorted([f.replace(".csv", "") for f in os.listdir("cache") if f.endswith(".csv")])
selected_chart_code = st.selectbox("🔍 Chọn mã để hiển thị biểu đồ kỹ thuật:", available_files)

if selected_chart_code:
    try:
        df_chart = pd.read_csv(f"cache/{selected_chart_code}.csv")

        # Chuẩn hóa cột thời gian
        if 'date' in df_chart.columns and 'time' not in df_chart.columns:
            df_chart.rename(columns={'date': 'time'}, inplace=True)
        df_chart['time'] = pd.to_datetime(df_chart['time'])
        df_chart = df_chart.sort_values("time")

        # Tính các chỉ báo kỹ thuật
        df_chart['MA10'] = df_chart['close'].rolling(10).mean()
        df_chart['MA20'] = df_chart['close'].rolling(20).mean()
        df_chart['MA50'] = df_chart['close'].rolling(50).mean()
        df_chart['MA100'] = df_chart['close'].rolling(100).mean()
        df_chart['MA200'] = df_chart['close'].rolling(200).mean()
        df_chart['EMA12'] = df_chart['close'].ewm(span=12, adjust=False).mean()
        df_chart['EMA26'] = df_chart['close'].ewm(span=26, adjust=False).mean()
        df_chart['MACD'] = df_chart['EMA12'] - df_chart['EMA26']
        df_chart['Signal'] = df_chart['MACD'].ewm(span=9, adjust=False).mean()
        delta = df_chart['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df_chart['RSI'] = 100 - (100 / (1 + rs))
        df_chart['20SD'] = df_chart['close'].rolling(window=20).std()
        df_chart['UpperBB'] = df_chart['MA20'] + 2 * df_chart['20SD']
        df_chart['LowerBB'] = df_chart['MA20'] - 2 * df_chart['20SD']

        # Tạo biểu đồ tổng hợp bằng Plotly
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            row_heights=[0.45, 0.2, 0.2, 0.15], vertical_spacing=0.03,
            subplot_titles=("Nến + MA/BB", "MACD", "RSI", "Volume")
        )

        # === Row 1: Nến + MA/BB ===
        fig.add_trace(go.Candlestick(
            x=df_chart['time'], open=df_chart['open'], high=df_chart['high'],
            low=df_chart['low'], close=df_chart['close'], name='Giá'
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart['time'], y=df_chart['MA10'], name='MA10', line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart['time'], y=df_chart['MA20'], name='MA20', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart['time'], y=df_chart['MA50'], name='MA50', line=dict(color='green')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart['time'], y=df_chart['MA100'], name='MA100', line=dict(color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart['time'], y=df_chart['MA200'], name='MA200', line=dict(color='purple')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart['time'], y=df_chart['UpperBB'], name='Upper BB', line=dict(color='gray', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart['time'], y=df_chart['LowerBB'], name='Lower BB', line=dict(color='gray', dash='dot')), row=1, col=1)

        # === Row 2: MACD ===
        fig.add_trace(go.Scatter(x=df_chart['time'], y=df_chart['MACD'], name='MACD', line=dict(color='purple')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_chart['time'], y=df_chart['Signal'], name='Signal', line=dict(color='red')), row=2, col=1)

        # === Row 3: RSI ===
        fig.add_trace(go.Scatter(x=df_chart['time'], y=df_chart['RSI'], name='RSI', line=dict(color='darkgreen')), row=3, col=1)
        fig.add_shape(type="line", x0=df_chart['time'].min(), x1=df_chart['time'].max(), y0=70, y1=70,
                      line=dict(color="red", dash="dot"), row=3, col=1)
        fig.add_shape(type="line", x0=df_chart['time'].min(), x1=df_chart['time'].max(), y0=30, y1=30,
                      line=dict(color="blue", dash="dot"), row=3, col=1)

        # === Row 4: Volume ===
        fig.add_trace(go.Bar(x=df_chart['time'], y=df_chart['volume'], name='Volume', marker_color='gray'), row=4, col=1)

        fig.update_layout(height=1000, title=f"Biểu đồ kỹ thuật tổng hợp: {selected_chart_code}", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Không thể hiển thị biểu đồ cho {selected_chart_code}: {e}")
