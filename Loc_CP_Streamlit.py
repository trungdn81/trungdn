import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(page_title="Lọc cổ phiếu tiềm năng theo quý", layout="wide")

# Dữ liệu mẫu đã đặt sẵn trong repo
EXCEL_FILE = "du_lieu_chung_khoan.xlsx"
FILTER_SAVE_FILE = "bo_loc_luu.json"

def load_data():
    return pd.read_excel(EXCEL_FILE, sheet_name=None)

def get_available_periods(data_dict):
    periods = set()
    for df in data_dict.values():
        if "period" in df.columns:
            periods.update(df["period"].dropna().astype(str).unique())
    return sorted(periods, reverse=True)

def apply_filters(df, filters):
    for f in filters:
        col = f['column']
        op = f['operator']
        val = f['value']
        try:
            val = float(val)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass
        if op == ">=":
            df = df[df[col] >= val]
        elif op == "<=":
            df = df[df[col] <= val]
        elif op == "==":
            df = df[df[col] == val]
        elif op == ">":
            df = df[df[col] > val]
        elif op == "<":
            df = df[df[col] < val]
    return df

def process_sheets(data_dict, selected_periods, filters, continue_from_previous, prev_results):
    result = prev_results.copy() if continue_from_previous else []
    for symbol, df in data_dict.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        try:
            df.columns = df.columns.str.strip()
            df["period"] = df["period"].astype(str)
            df = df[df["period"].isin(selected_periods)]
            if df.empty:
                continue
            df_filtered = apply_filters(df.copy(), filters)
            if not df_filtered.empty:
                result.append(symbol)
        except:
            continue
    return result
# Giao diện
st.title("📊 Lọc cổ phiếu tiềm năng theo quý")

data_dict = load_data()
available_periods = get_available_periods(data_dict)

# ========== SIDEBAR ==========
st.sidebar.header("🎛️ Cấu hình lọc")

selected_periods = st.sidebar.multiselect("Chọn các quý:", available_periods, default=available_periods[:1])

with st.sidebar.expander("➕ Thêm điều kiện lọc"):
    # Gộp tất cả các cột số để người dùng chọn
    columns = sorted(list({col for df in data_dict.values() for col in df.columns if df[col].dtype != 'O'}))
    new_filter_col = st.selectbox("Chọn cột:", columns)
    new_filter_op = st.selectbox("Toán tử:", [">=", "<=", "==", ">", "<"])
    new_filter_val = st.text_input("Giá trị:", value="0")
    add_filter = st.button("➕ Thêm điều kiện")

# Bộ nhớ session để lưu điều kiện lọc
if "filters" not in st.session_state:
    st.session_state["filters"] = []

if add_filter:
    st.session_state.filters.append({
        "column": new_filter_col,
        "operator": new_filter_op,
        "value": new_filter_val
    })

# Hiển thị danh sách bộ lọc
if st.session_state.filters:
    st.sidebar.markdown("### 📋 Danh sách bộ lọc hiện tại:")
    for i, f in enumerate(st.session_state.filters):
        st.sidebar.write(f"{i+1}. {f['column']} {f['operator']} {f['value']}")

    if st.sidebar.button("🗑 Xóa tất cả bộ lọc"):
        st.session_state.filters = []
# ======= MAIN =========
st.markdown("## 🚦 Kết quả lọc")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🆕 Lọc mới"):
        result = process_sheets(data_dict, selected_periods, st.session_state.filters, False, [])
        st.session_state["ket_qua_loc"] = result
with col2:
    if st.button("🔄 Lọc tiếp"):
        previous = st.session_state.get("ket_qua_loc", [])
        result = process_sheets(data_dict, selected_periods, st.session_state.filters, True, previous)
        st.session_state["ket_qua_loc"] = result
with col3:
    if st.button("💾 Lưu bộ lọc"):
        with open(FILTER_SAVE_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.filters, f)
        st.success("✅ Đã lưu bộ lọc vào file.")

# 📂 Tải lại bộ lọc đã lưu
if os.path.exists(FILTER_SAVE_FILE):
    if st.button("📂 Tải bộ lọc đã lưu"):
        with open(FILTER_SAVE_FILE, "r", encoding="utf-8") as f:
            st.session_state.filters = json.load(f)
        st.success("✅ Đã tải bộ lọc từ file.")

# ✅ Hiển thị kết quả
ket_qua = st.session_state.get("ket_qua_loc", [])
st.write(f"🔎 Số mã cổ phiếu được lọc: **{len(ket_qua)}**")
if ket_qua:
    df_result = pd.DataFrame(ket_qua, columns=["Mã cổ phiếu"])
    st.dataframe(df_result)

    # 📥 Xuất kết quả ra Excel
    import io
    with io.BytesIO() as output:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_result.to_excel(writer, index=False)
        st.download_button(
            "📥 Tải kết quả Excel",
            data=output.getvalue(),
            file_name="co_phieu_loc.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
