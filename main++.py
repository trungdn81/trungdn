# 📁 File: main.py (phiên bản mới với 4 tab)

import streamlit as st
import pandas as pd
import json
import os
import io
from datetime import datetime, timedelta
from vnstock import Vnstock, Quote

st.set_page_config(page_title="Định giá cổ phiếu - WebApp", layout="wide")

TRONG_SO_MAC_DINH = {"P/E": 0.4, "P/B": 0.25, "ROE": 0.15, "PEG": 0.1, "DCF": 0.1}
EXCEL_FILE = "du_lieu_chung_khoan.xlsx"
GIA1_FILE = "gia_CP.xlsx"
GIA0_FILE = "gia_CP(back_test).xlsx"
FILTER_SAVE_FILE = "bo_loc_luu.json"

# ======= Các hàm chung =======
def normalize(text):
    import re
    return re.sub(r"[^a-zA-Z0-9]", "", str(text)).lower()

def get_value(row, keywords):
    norm_row = {normalize(str(k)): str(k) for k in row.keys()}
    for kw in keywords:
        norm_kw = normalize(kw)
        if norm_kw in norm_row:
            try:
                val = row[norm_row[norm_kw]]
                val_str = str(val).strip()
                if val_str in ["", "NA", "N/A", "--", "None"]:
                    return None
                return float(val_str.replace(",", ""))
            except:
                continue
    return None

def dinh_gia(eps, pe, pb, bvps, roe):
    res = {}
    if eps and pe:
        res["P/E"] = eps * pe
    if pb and bvps:
        res["P/B"] = pb * bvps
    if roe and bvps:
        res["ROE"] = bvps * roe / 0.13
    if eps:
        res["PEG"] = eps * 0.1 * 100
        g, r, n = 0.12, 0.15, 5
        res["DCF"] = sum([(eps * (1 + g) ** i) / (1 + r) ** i for i in range(1, n + 1)])
    return res

def run_backtest(bctc_file, gia_file0, gia_file1):
    bctc_data = pd.read_excel(bctc_file, sheet_name=None)
    gia_T0_dict = dict(zip(pd.read_excel(gia_file0)["symbol"], pd.read_excel(gia_file0)["close_price"]))
    gia_T1_dict = dict(zip(pd.read_excel(gia_file1)["symbol"], pd.read_excel(gia_file1)["close_price"]))

    results = []
    for symbol, df in bctc_data.items():
        try:
            df.columns = df.columns.str.strip()
            if "period" in df.index.names:
                df = df.reset_index()
            df["period"] = df["period"].astype(str)
            df = df.sort_values("period", ascending=False)
            latest = df.iloc[0]

            eps = get_value(latest, ["EPS", "earning_per_share"])
            pe = get_value(latest, ["P/E", "price_to_earning"])
            pb = get_value(latest, ["P/B", "price_to_book"])
            roe = get_value(latest, ["ROE", "return_on_equity"])
            bvps = get_value(latest, ["book_value_per_share", "BVPS", "giá trị sổ sách", "giá trị sổ sách/cổ phiếu"])

            gia_T0 = gia_T0_dict.get(symbol)
            gia_T1 = gia_T1_dict.get(symbol)
            if gia_T0 is None or gia_T1 is None:
                continue

            fair_prices = dinh_gia(eps, pe, pb, bvps, roe)
            for method, fair in fair_prices.items():
                signal = "MUA" if fair > gia_T0 * 1.1 else ("BAN" if fair < gia_T0 * 0.9 else "GIU")
                thuc_te = "TANG" if gia_T1 > gia_T0 else "GIAM"
                is_correct = (signal == "MUA" and thuc_te == "TANG") or (signal == "BAN" and thuc_te == "GIAM")

                results.append({
                    "symbol": symbol,
                    "method": method,
                    "gia_T0": round(gia_T0, 2),
                    "gia_T1": round(gia_T1, 2),
                    "dinh_gia": round(fair, 2),
                    "tín_hiệu": signal,
                    "thực_tế": thuc_te,
                    "đúng": is_correct
                })
        except:
            continue

    df_result = pd.DataFrame(results)
    if df_result.empty:
        return None, None, None

    summary = df_result.groupby("method")["đúng"].agg(["count", "sum"])
    summary["accuracy_%"] = summary["sum"] / summary["count"] * 100
    summary = summary.rename(columns={"sum": "đúng", "count": "tổng"}).reset_index()

    correct_all5 = df_result[df_result["đúng"] == True]
    pivot = correct_all5.pivot_table(index="symbol", columns="method", values="đúng", aggfunc="first")
    common_symbols = pivot.dropna().index.tolist()

    df_top_all = df_result[df_result["symbol"].isin(common_symbols)]
    df_top_all = df_top_all.groupby("symbol").filter(lambda x: len(x) == 5 and all(x["tín_hiệu"] == "MUA") and all(x["thực_tế"] == "TANG"))

    return df_result, summary, df_top_all

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

# Tiếp theo: tạo 4 tab và chia nội dung theo yêu cầu...

# ==== 4 TAB CHÍNH ====
tab1, tab2, tab3, tab4 = st.tabs([
    "📌 Định giá riêng lẻ",
    "📈 Backtest",
    "🏆 Lọc Top định giá",
    "📊 Lọc cổ phiếu tiềm năng"
])

# ===== TAB 1: Định giá riêng lẻ =====
with tab1:
    st.subheader("📌 Định giá cổ phiếu bằng mã (API vnstock)")
    symbol = st.text_input("Nhập mã cổ phiếu (ví dụ: FPT, VNM, DPG)", value="FPT")
    trong_so = {}
    st.markdown("### 📊 Trọng số:")
    cols = st.columns(5)
    for i, key in enumerate(TRONG_SO_MAC_DINH):
        with cols[i]:
            trong_so[key] = st.number_input(f"{key}", min_value=0.0, max_value=1.0,
                                            value=TRONG_SO_MAC_DINH[key], step=0.05, key=f"ts_{key}")

    if st.button("🔍 Thực hiện định giá"):
        try:
            stock_data = Vnstock().stock(symbol=symbol, source='TCBS')
            df = stock_data.finance.ratio(period="quarter")
            df.columns = df.columns.str.strip()
            if "period" in df.index.names:
                df = df.reset_index()
            if "period" not in df.columns:
                df["period"] = df.index.astype(str)
            df = df.sort_values("period", ascending=False)
            latest = df.iloc[0]

            eps = get_value(latest, ["EPS", "earning_per_share"])
            pe = get_value(latest, ["P/E", "price_to_earning"])
            pb = get_value(latest, ["P/B", "price_to_book"])
            roe = get_value(latest, ["ROE", "return_on_equity"])
            bvps = get_value(latest, ["book_value_per_share", "BVPS", "giá trị sổ sách", "giá trị sổ sách/cổ phiếu"])

            fair = dinh_gia(eps, pe, pb, bvps, roe)
            st.markdown("### 📈 Kết quả định giá từng phương pháp:")
            for method, val in fair.items():
                st.write(f"- **{method}**: {val:,.2f} VND")

            ts_sum = sum(trong_so[m] for m in fair if m in trong_so)
            total = sum(fair[m] * trong_so[m] for m in fair if m in trong_so)
            gia_dinh_gia = total / ts_sum if ts_sum else None

            st.markdown(f"### 🎯 Giá trị định giá trung bình theo trọng số: **{gia_dinh_gia:,.2f} VND**")

            try:
                quote = Quote(symbol=symbol, source="VCI")
                realtime_data = quote.realtime()
                price_raw = realtime_data.get("priceMatched") or realtime_data.get("matchedPrice") or 0
                market_price = float(price_raw) * 1000 if price_raw else None
            except:
                try:
                    today = datetime.now().date()
                    start = (today - timedelta(days=7)).strftime("%Y-%m-%d")
                    end = today.strftime("%Y-%m-%d")
                    df_price = quote.history(start=start, end=end)
                    df_price = df_price.dropna(subset=["close"])
                    market_price = df_price.iloc[-1]["close"] * 1000
                except:
                    market_price = None

            if market_price:
                st.markdown(f"### 💰 Giá thị trường hiện tại: **{market_price:,.2f} VND**")
                chenh_lech = gia_dinh_gia - market_price
                ty_le = chenh_lech / market_price * 100
                st.markdown(f"### 🧮 Chênh lệch: **{chenh_lech:,.0f} VND** ({ty_le:.2f}%)")
                if ty_le > 10:
                    st.success("✅ Khuyến nghị: **NÊN MUA**")
                elif ty_le < -10:
                    st.error("⚠️ Khuyến nghị: **NÊN BÁN**")
                else:
                    st.info("⏸ Khuyến nghị: **GIỮ**")
            else:
                st.warning("⚠️ Không lấy được giá thị trường.")

        except Exception as e:
            st.error(f"❌ Lỗi xử lý: {e}")
# ===== TAB 2: Backtest thị trường =====
with tab2:
    st.subheader("📈 Backtest định giá từ dữ liệu có sẵn trong repo")

    if st.button("🚀 Chạy Backtest"):
        with st.spinner("Đang xử lý dữ liệu..."):
            try:
                df_result, summary, df_top_all = run_backtest(EXCEL_FILE, GIA0_FILE, GIA1_FILE)

                if df_result is None:
                    st.warning("⚠️ Không có dữ liệu kết quả.")
                else:
                    st.success("✅ Đã xử lý dữ liệu mẫu thành công")

                    st.markdown("### 📊 Kết quả tổng hợp")
                    st.dataframe(summary)

                    st.markdown("### 📋 Chi tiết từng dòng")
                    st.dataframe(df_result.head(100))

                    st.markdown("### 🏆 Top cổ phiếu đúng cả 5 phương pháp & tăng giá")
                    st.dataframe(df_top_all)

                    # Xuất ra Excel
                    with io.BytesIO() as output:
                        with pd.ExcelWriter(output, engine="openpyxl") as writer:
                            df_result.to_excel(writer, sheet_name="Chi_tiet", index=False)
                            summary.to_excel(writer, sheet_name="Tong_hop", index=False)
                            df_top_all.to_excel(writer, sheet_name="Top_chinh_xac", index=False)
                        st.download_button("📥 Tải kết quả Backtest",
                            data=output.getvalue(),
                            file_name="ket_qua_backtest.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            except Exception as e:
                st.error(f"❌ Lỗi khi đọc dữ liệu mẫu: {e}")
# ===== TAB 3: Lọc Top cổ phiếu theo định giá =====
with tab3:
    st.subheader("🏆 Lọc Top cổ phiếu theo định giá từ dữ liệu có sẵn")

    top_n = st.selectbox("Số lượng mã muốn lọc", [10, 20, 50, 100], index=0)
    ts_top = {}
    col1, col2 = st.columns(2)
    with col1:
        for key in TRONG_SO_MAC_DINH:
            ts_top[key] = st.number_input(f"Trọng số {key}", min_value=0.0, max_value=1.0,
                                          value=TRONG_SO_MAC_DINH[key], step=0.01, key=f"top_{key}")

    if st.button("📊 Lọc Top"):
        results = []
        try:
            all_sheets = pd.read_excel(EXCEL_FILE, sheet_name=None)
            for sheet_name, df in all_sheets.items():
                try:
                    df.columns = df.columns.str.strip()
                    if "period" in df.index.names:
                        df = df.reset_index()
                    df["period"] = df["period"].astype(str)
                    df = df.sort_values("period", ascending=False)
                    latest = df.iloc[0]

                    eps = get_value(latest, ["EPS", "earning_per_share"])
                    pe = get_value(latest, ["P/E", "price_to_earning"])
                    pb = get_value(latest, ["P/B", "price_to_book"])
                    roe = get_value(latest, ["ROE", "return_on_equity"])
                    bvps = get_value(latest, ["book_value_per_share", "BVPS", "giá trị sổ sách", "giá trị sổ sách/cổ phiếu"])

                    fair = dinh_gia(eps, pe, pb, bvps, roe)
                    final = 0
                    ts_sum = 0
                    for k, val in fair.items():
                        if k in ts_top:
                            final += val * ts_top[k]
                            ts_sum += ts_top[k]
                    if ts_sum:
                        results.append((sheet_name, final / ts_sum))
                except:
                    continue

            results = sorted(results, key=lambda x: x[1], reverse=True)[:top_n]
            df_top = pd.DataFrame(results, columns=["Mã cổ phiếu", "Định giá"])
            st.dataframe(df_top)

            # Xuất kết quả ra Excel
            with io.BytesIO() as output:
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    df_top.to_excel(writer, index=False)
                st.download_button("📥 Tải danh sách Top cổ phiếu",
                                   data=output.getvalue(),
                                   file_name="top_dinh_gia.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            st.error(f"❌ Lỗi đọc file mẫu: {e}")
# ===== TAB 4: Lọc cổ phiếu tiềm năng theo quý =====
with tab4:
    st.subheader("📊 Lọc cổ phiếu tiềm năng theo quý")

    data_dict = load_data()
    available_periods = get_available_periods(data_dict)

    selected_periods = st.multiselect("Chọn các quý:", available_periods, default=available_periods[:1])

    with st.expander("➕ Thêm điều kiện lọc"):
        columns = sorted(list({col for df in data_dict.values() for col in df.columns if df[col].dtype != 'O'}))
        new_filter_col = st.selectbox("Chọn cột:", columns)
        new_filter_op = st.selectbox("Toán tử:", [">=", "<=", "==", ">", "<"])
        new_filter_val = st.text_input("Giá trị:", value="0")
        add_filter = st.button("➕ Thêm điều kiện")

    if "filters" not in st.session_state:
        st.session_state["filters"] = []

    if add_filter:
        st.session_state.filters.append({
            "column": new_filter_col,
            "operator": new_filter_op,
            "value": new_filter_val
        })

    if st.session_state.filters:
        st.markdown("### 📋 Danh sách bộ lọc hiện tại:")
        for i, f in enumerate(st.session_state.filters):
            st.write(f"{i+1}. {f['column']} {f['operator']} {f['value']}")

        if st.button("🗑 Xóa tất cả bộ lọc"):
            st.session_state.filters = []

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

    if os.path.exists(FILTER_SAVE_FILE):
        if st.button("📂 Tải bộ lọc đã lưu"):
            with open(FILTER_SAVE_FILE, "r", encoding="utf-8") as f:
                st.session_state.filters = json.load(f)
            st.success("✅ Đã tải bộ lọc từ file.")

    ket_qua = st.session_state.get("ket_qua_loc", [])
    st.write(f"🔎 Số mã cổ phiếu được lọc: **{len(ket_qua)}**")
    if ket_qua:
        df_result = pd.DataFrame(ket_qua, columns=["Mã cổ phiếu"])
        st.dataframe(df_result)

        with io.BytesIO() as output:
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_result.to_excel(writer, index=False)
            st.download_button(
                "📥 Tải kết quả Excel",
                data=output.getvalue(),
                file_name="co_phieu_loc.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
