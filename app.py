import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from vnstock import Vnstock  # <-- Đã thêm đúng hàm cần dùng
from datetime import datetime, timedelta
import time
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Tạo thư mục cache nếu chưa tồn tại
if not os.path.exists("cache"):
    os.makedirs("cache")

# ==== Thông tin logic chỉ báo ==== #
logic_info = {
    "MA20": {"mota": "Giá hiện tại cắt lên MA20", "vaitro": "Xác định xu hướng ngắn hạn", "uutien": "⭐⭐⭐"},
    "MACD": {"mota": "MACD > Signal và cắt lên", "vaitro": "Tín hiệu đảo chiều", "uutien": "⭐⭐⭐⭐"},
    "RSI": {"mota": "RSI từ 50–80", "vaitro": "Đo sức mạnh giá", "uutien": "⭐⭐"},
    "BB": {"mota": "Giá vượt dải BB trên", "vaitro": "Breakout", "uutien": "⭐"},
    "ADX": {"mota": "ADX > 20", "vaitro": "Xác nhận xu hướng mạnh", "uutien": "⭐⭐"},
}

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

def update_price_cache(symbol, start_date, end_date, source="VCI"):
    try:
        path = f"cache/{symbol}.csv"
        if os.path.exists(path):
            old_df = pd.read_csv(path)
            if 'date' in old_df.columns and 'time' not in old_df.columns:
                old_df.rename(columns={'date': 'time'}, inplace=True)
            old_df['time'] = pd.to_datetime(old_df['time'])
            last_date = old_df['time'].max()
            fetch_start_dt = last_date + timedelta(days=1)
        else:
            old_df = pd.DataFrame()
            fetch_start_dt = datetime.strptime(start_date, "%Y-%m-%d")

        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        if fetch_start_dt > end_dt:
            print(f"✅ {symbol} đã đầy đủ dữ liệu.")
            return

        # Dùng Vnstock kiểu OOP để lấy dữ liệu
        stock = Vnstock().stock(symbol=symbol, source=source)
        new_df = stock.quote.history(start=fetch_start_dt.strftime("%Y-%m-%d"), end=end_date)

        new_df['time'] = pd.to_datetime(new_df['time'])
        full_df = pd.concat([old_df, new_df]).drop_duplicates(subset=["time"]).sort_values("time")
        full_df.to_csv(path, index=False)
        print(f"✅ Cập nhật dữ liệu giá cho {symbol} thành công.")
    except Exception as e:
        print(f"❌ Lỗi khi cập nhật giá cho {symbol}: {e}")
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

# Khởi tạo session state
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
def safe_int(val, default):  # int
    try: return int(val)
    except: return default

def safe_float(val, default):  # float
    try: return float(val)
    except: return default

# Chọn preset điều kiện lọc
selected_preset = st.sidebar.selectbox("Chọn preset chiến lược", list(extra_conditions.keys()))
if st.sidebar.button("🔄 Áp dụng điều kiện lọc"):
    st.session_state['preset_conditions'] = extra_conditions[selected_preset]
preset_condition = st.session_state.get('preset_conditions', extra_conditions[selected_preset])

# Giao diện lọc
price_min = st.sidebar.number_input("Giá tối thiểu (VND)", min_value=0, max_value=1_000_000, value=safe_int(preset_condition.get("price_min"), 0), step=1000)
price_max = st.sidebar.number_input("Giá tối đa (VND)", min_value=0, max_value=1_000_000, value=safe_int(preset_condition.get("price_max"), 200000), step=1000)
price_change_5d_min = st.sidebar.slider("Tăng giá 5 phiên gần nhất tối thiểu (%)", -20, 20, value=safe_int(preset_condition.get("price_change_5d_min"), -20))
price_change_5d_max = st.sidebar.slider("Tăng giá 5 phiên gần nhất tối đa (%)", -20, 20, value=safe_int(preset_condition.get("price_change_5d_max"), 20))
volume_ratio_min = st.sidebar.slider("Volume hôm nay lớn hơn bao nhiêu lần TB20", 0.0, 5.0, step=0.1, value=safe_float(preset_condition.get("volume_ratio_min"), 0.0))
rsi_min = st.sidebar.slider("RSI tối thiểu", 0, 100, value=safe_int(preset_condition.get("rsi_min"), 0))

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

# ==== Giao diện chính: Bộ lọc kỹ thuật ==== #
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
        result, logic_counts = [], {k: 0 for k in logic_info}
        cache_dir = "cache"
        if os.path.exists(cache_dir):
            files = [f for f in os.listdir(cache_dir) if f.endswith(".csv")]
            for file in files:
                symbol = file.replace(".csv", "")
                try:
                    df = pd.read_csv(os.path.join(cache_dir, file))
                    # df = pd.read_csv(url)
                    # url = 'https://raw.githubusercontent.com/trungdn81/dinh_gia-trungdn-/main/danh_sach_ma.csv'
                    if 'date' in df.columns and 'time' not in df.columns:
                        df.rename(columns={'date': 'time'}, inplace=True)
                    df['time'] = pd.to_datetime(df['time'])

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

                    # Tính điểm
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

                    score, note, logic = score_stock(df, {k: st.session_state[f"selected_{k}"] for k in logic_info}, {k: st.session_state[f"weight_{k}"] for k in logic_info})
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
        else:
            st.warning("❗ Không có mã nào đạt đủ điều kiện.")

# ==== Hiển thị biểu đồ kỹ thuật của bất kỳ mã nào ==== #
st.markdown("## 📈 Xem biểu đồ kỹ thuật cho mã bất kỳ")

# Đọc toàn bộ file cache để chọn mã
available_files = sorted([f.replace(".csv", "") for f in os.listdir("cache") if f.endswith(".csv")])
selected_chart_code = st.selectbox("🔍 Chọn mã để hiển thị biểu đồ kỹ thuật:", available_files)

if selected_chart_code:
    try:
        df_chart = pd.read_csv(f"cache/{selected_chart_code}.csv")

        # Xử lý cột time nếu tên là 'date'
        if 'date' in df_chart.columns and 'time' not in df_chart.columns:
            df_chart.rename(columns={'date': 'time'}, inplace=True)
        df_chart['time'] = pd.to_datetime(df_chart['time'])

        # Tính toán các chỉ báo
        df_chart = df_chart.sort_values("time")
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

        # Tạo biểu đồ tổng hợp
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

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


        # ✅ Xuất Excel nếu có dữ liệu
        if 'df_result' in locals() and (not df_result.empty or not logic_df.empty):
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
