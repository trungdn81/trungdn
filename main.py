import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import re
from vnstock import Vnstock, Quote

st.set_page_config(page_title="Định giá cổ phiếu 3 tab", layout="wide")

TRONG_SO_MAC_DINH = {"P/E": 0.4, "P/B": 0.25, "ROE": 0.15, "PEG": 0.1, "DCF": 0.1}

def normalize(text):
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
        res["DCF"] = sum([(eps * (1 + g)**i) / (1 + r)**i for i in range(1, n + 1)])
    return res
# ===== Giao diện 3 tab =====
tab1, tab2, tab3 = st.tabs(["📌 Định giá riêng lẻ (API)", "📈 Backtest", "🏆 Lọc Top"])

with tab1:
    st.subheader("📘 Định giá cổ phiếu bằng mã (API vnstock)")

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
with tab2:
    st.subheader("📈 Backtest định giá từ dữ liệu có sẵn trong repo")

    if st.button("🚀 Chạy Backtest"):
        with st.spinner("Đang xử lý dữ liệu..."):
            try:
                # Sử dụng các file Excel đã lưu sẵn trong repo
                bctc_file = "du_lieu_chung_khoan.xlsx"
                gia1_file = "gia_CP.xlsx"
                gia0_file = "gia_CP(back_test).xlsx"

                df_result, summary, df_top_all = run_backtest(bctc_file, gia0_file, gia1_file)

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

            except Exception as e:
                st.error(f"❌ Lỗi khi đọc dữ liệu mẫu: {e}")
with tab3:
    st.subheader("🏆 Lọc Top cổ phiếu theo định giá từ file có sẵn")

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
            # Đọc file BCTC nhiều mã (mỗi sheet là một mã)
            all_sheets = pd.read_excel("du_lieu_chung_khoan.xlsx", sheet_name=None)
            for sheet_name, df in all_sheets.items():
                try:
                    df.columns = df.columns.str.strip()
                    if "period" in df.index.names:
                        df = df.reset_index()
                    df["period"] = df["period"].astype(str)
                    df = df.sort_values("period", ascending=False)
                    latest = df.iloc[0]

                    eps = get_value(latest, ["eps", "EPS", "Earnings Per Share"])
                    pe = get_value(latest, ["pe", "p/e", "P/E"])
                    pb = get_value(latest, ["pb", "p/b", "P/B"])
                    roe = get_value(latest, ["roe", "ROE"])
                    bvps = get_value(latest, ["book_value_per_share", "BVPS", "Giá trị sổ sách"])

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

        except Exception as e:
            st.error(f"❌ Lỗi đọc file mẫu: {e}")
