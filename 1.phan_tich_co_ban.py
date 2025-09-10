# 📁 File: main_final.py (đã chỉnh sửa — Tab 3 xuất dữ liệu định giá toàn bộ + so sánh giá thị trường)

import streamlit as st
import pandas as pd
import json
import os
import io
import re
import unicodedata
import time
from vnstock import Vnstock
from datetime import datetime, timedelta
from vnstock import Vnstock, Quote

st.set_page_config(page_title="Định giá cổ phiếu - WebApp", layout="wide")

TRONG_SO_MAC_DINH = {"P/E": 0.5, "P/B": 0.5, "ROE": 0.0, "PEG": 0.0, "DCF": 0.0}
EXCEL_FILE = "du_lieu_chung_khoan.xlsx"
GIA1_FILE = "gia_T1.xlsx"
GIA0_FILE = "gia_T0.xlsx"
FILTER_SAVE_FILE = "bo_loc_luu.json"

# ======= Các hàm chung =======
def normalize(text):
    return re.sub(r"[^a-zA-Z0-9]", "", str(text)).lower()

def get_value(row, keywords):
    try:
        norm_row = {normalize(str(k)): str(k) for k in row.keys()}
    except Exception:
        return None
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

# ===== Hàm định giá chuẩn dùng chung cho Tab1 và Tab2 =====
def dinh_gia(eps, pe, pb, bvps, roe, params=None):
    """
    Hàm định giá chuẩn cho cổ phiếu theo nhiều phương pháp cơ bản:
    - P/E: EPS * P/E
    - P/B: BVPS * P/B
    - ROE: BVPS * ROE / Cost_of_Equity
    - PEG: EPS * Growth%   (PEG hợp lý = 1)
    - DCF: Dòng tiền dự phóng n năm + Terminal Value

    Đầu vào:
        eps   : EPS (lợi nhuận trên mỗi cổ phiếu, dạng float)
        pe    : P/E hiện tại
        pb    : P/B hiện tại
        bvps  : Giá trị sổ sách / cổ phiếu
        roe   : ROE (% hoặc decimal đều được, sẽ tự chuẩn hóa)
        params: dict gồm các tham số:
            - dcf_g (mặc định 0.12 = 12%)
            - dcf_r (mặc định 0.15 = 15%)
            - dcf_n (mặc định 5 năm)
            - peg_target (mặc định 1.0)
            - cost_of_equity (mặc định 0.1 = 10%)
    """

    # --- Hàm phụ xử lý an toàn ---
    def _to_float_safe(x):
        try:
            if x is None:
                return None
            if isinstance(x, str):
                x = x.replace(",", "").strip()
            return float(x)
        except:
            return None

    def _normalize_ratio(x):
        """Chuẩn hóa tỷ lệ %: nếu >2 thì chia 100 (ví dụ 8.8 -> 0.088)"""
        try:
            v = float(x)
            return v / 100 if v > 2 else v
        except:
            return None

    def _normalize_growth(g):
        """Chuẩn hóa tăng trưởng: nếu >2 thì chia 100 (ví dụ 12 -> 0.12)"""
        try:
            v = float(g)
            return v / 100 if v > 2 else v
        except:
            return None

    # --- Chuẩn hóa inputs ---
    eps_v  = _to_float_safe(eps)
    pe_v   = _to_float_safe(pe)
    pb_v   = _to_float_safe(pb)
    bvps_v = _to_float_safe(bvps)
    roe_v  = _normalize_ratio(roe)

    if params is None:
        params = {}

    growth_v = _normalize_growth(params.get("growth", None) or params.get("dcf_g", None))

    # --- Kết quả ---
    res = {}

    # P/E
    try:
        if eps_v is not None and pe_v is not None:
            res["P/E"] = eps_v * pe_v
        else:
            res["P/E"] = None
    except:
        res["P/E"] = None

    # P/B
    try:
        if bvps_v is not None and pb_v is not None:
            res["P/B"] = bvps_v * pb_v
        else:
            res["P/B"] = None
    except:
        res["P/B"] = None

    # ROE
    try:
        cost_of_equity = float(params.get("cost_of_equity", 0.1))
        if bvps_v is not None and roe_v is not None and cost_of_equity > 0:
            res["ROE"] = bvps_v * roe_v / cost_of_equity
        else:
            res["ROE"] = None
    except:
        res["ROE"] = None

    # PEG (PEG=1 => P/E hợp lý = Growth%)
    try:
        peg_target = float(params.get("peg_target", 1.0))
        if eps_v is not None and growth_v is not None and growth_v > 0:
            fair_pe = peg_target * (growth_v * 100)  # growth=0.12 -> fair_PE=12
            res["PEG"] = eps_v * fair_pe
        else:
            res["PEG"] = None
    except:
        res["PEG"] = None

    # DCF nhiều kỳ + Terminal Value
    try:
        if eps_v is not None:
            g = _normalize_growth(params.get("dcf_g", growth_v if growth_v else 0.12))
            r = float(params.get("dcf_r", 0.15))
            n = int(params.get("dcf_n", 5))

            pv_sum = 0.0
            for i in range(1, n+1):
                cf_i = eps_v * (1 + g) ** i
                pv_sum += cf_i / (1 + r) ** i

            # Terminal Value nếu r > g
            if r > g and g is not None:
                cf_n = eps_v * (1 + g) ** n
                tv = cf_n * (1 + g) / (r - g)
                pv_sum += tv / (1 + r) ** n

            res["DCF"] = pv_sum
        else:
            res["DCF"] = None
    except:
        res["DCF"] = None

    return res


# ==== Helpers period ====
period_pattern = re.compile(r'^\s*(\d{4})\s*[-/_ ]?\s*[Qq](\d)\s*$', re.IGNORECASE)

def parse_period_tuple(s):
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    m = period_pattern.match(s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    nums = re.findall(r'\d+', s)
    if len(nums) >= 2:
        try:
            year = int(nums[0])
            q = int(nums[1]) if len(nums[1]) == 1 else int(nums[1][-1])
            return (year, q)
        except:
            return (0, 0)
    return (0, 0)

def period_to_sort_value(s):
    y, q = parse_period_tuple(s)
    return y * 10 + q

def detect_period_col(df):
    for col in df.columns:
        cn = str(col).lower()
        if 'back to menu' in cn or ('back' in cn and 'menu' in cn) or 'quý' in cn or 'kỳ' in cn or 'ky' in cn or 'period' in cn:
            return col
    for col in df.columns:
        try:
            sample_vals = df[col].dropna().astype(str).head(10)
        except:
            continue
        if any(period_pattern.match(str(v)) for v in sample_vals):
            return col
    if len(df.columns) > 0:
        try:
            sample_vals = df[df.columns[0]].dropna().astype(str).head(10)
            if any(period_pattern.match(str(v)) for v in sample_vals):
                return df.columns[0]
        except:
            pass
    return None

# ==== Load data từ Excel ====
def load_data():
    data = pd.read_excel(EXCEL_FILE, sheet_name=None)
    normalized = {}
    rename_map_basic = {
        "Giá trị sổ sách/cổ phiếu": "BVPS",
        "Giá trị sổ sách": "BVPS",
        "EPS cơ bản": "EPS",
        "Quý": "period",
        "Kỳ": "period",
        "Kỳ ": "period",
    }
    for sheet, df in data.items():
        try:
            df = df.copy()
            df.columns = df.columns.str.strip()
            df = df.rename(columns=rename_map_basic)
            pcol = detect_period_col(df)
            if pcol is not None and pcol != "period":
                df = df.rename(columns={pcol: "period"})
            if "period" not in df.columns:
                df = df.reset_index()
                pcol2 = detect_period_col(df)
                if pcol2 is not None and pcol2 != "period":
                    df = df.rename(columns={pcol2: "period"})
            if "period" not in df.columns:
                df["period"] = df.index.astype(str)
            df["period"] = df["period"].astype(str)
            normalized[sheet] = df
        except Exception:
            continue
    return normalized

def get_available_periods(data_dict):
    periods = set()
    for df in data_dict.values():
        if "period" in df.columns:
            vals = df["period"].dropna().astype(str).unique().tolist()
            for v in vals:
                if period_pattern.match(v):
                    periods.add(v)
    sorted_periods = sorted(list(periods), key=period_to_sort_value, reverse=True)
    return sorted_periods

def apply_filters(df, filters):
    for f in filters:
        col = f['column']
        op = f['operator']
        val = f['value']
        try:
            val_num = float(val)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            val_num = val
        try:
            if op == ">=":
                df = df[df[col] >= val_num]
            elif op == "<=":
                df = df[df[col] <= val_num]
            elif op == "==":
                df = df[df[col] == val_num]
            elif op == ">":
                df = df[df[col] > val_num]
            elif op == "<":
                df = df[df[col] < val_num]
        except Exception:
            continue
    return df

def process_sheets(data_dict, selected_periods, filters, continue_from_previous, prev_results):
    result = prev_results.copy() if continue_from_previous else []
    for symbol, df in data_dict.items():
        try:
            df.columns = df.columns.str.strip()
            if "period" not in df.columns:
                continue
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

# ==== Backtest ====
def run_backtest(bctc_file, gia_file0, gia_file1):
    bctc_data = load_data()
    try:
        gia0 = pd.read_excel(gia_file0)
        gia1 = pd.read_excel(gia_file1)
        gia_T0_dict = dict(zip(gia0["symbol"], gia0["close_price"]))
        gia_T1_dict = dict(zip(gia1["symbol"], gia1["close_price"]))
    except:
        gia_T0_dict = {}
        gia_T1_dict = {}

    results = []
    for symbol, df in bctc_data.items():
        try:
            if "period" not in df.columns:
                continue
            if not any(period_pattern.match(str(x)) for x in df["period"].dropna().astype(str).unique()):
                continue
            df["__pnum"] = df["period"].apply(period_to_sort_value)
            latest = df.loc[df["__pnum"].idxmax()]

            eps = get_value(latest, ["EPS", "earning_per_share"])
            pe = get_value(latest, ["P/E", "price_to_earning"])
            pb = get_value(latest, ["P/B", "price_to_book"])
            roe = get_value(latest, ["ROE", "return_on_equity"])
            bvps = get_value(latest, ["BVPS", "book_value_per_share", "giá trị sổ sách", "giá trị sổ sách/cổ phiếu"])

            gia_T0 = gia_T0_dict.get(symbol)
            gia_T1 = gia_T1_dict.get(symbol)
            if gia_T0 is None or gia_T1 is None:
                continue

            fair_prices = dinh_gia(eps, pe, pb, bvps, roe)
            for method, fair in fair_prices.items():
                try:
                    # Nếu fair value quá lớn so với giá thị trường, chia xuống cùng đơn vị với T0
                    if fair is None:
                        signal = "GIU"
                    else:
                        # Tự động scale: nếu fair >> T0, giả sử T0 theo nghìn đồng
                        if fair > gia_T0 * 100:
                            fair_scaled = fair / 1000
                        else:
                            fair_scaled = fair

                        # Tính tín hiệu dựa trên giá đã scale
                        signal = (
                            "MUA" if fair_scaled > gia_T0 * 1.1
                            else ("BAN" if fair_scaled < gia_T0 * 0.9 else "GIU")
                        )

                    # Thực tế tăng giảm
                    thuc_te = "TANG" if gia_T1 > gia_T0 else "GIAM"

                    # Kiểm tra đúng/sai
                    is_correct = (signal == "MUA" and thuc_te == "TANG") or \
                                 (signal == "BAN" and thuc_te == "GIAM")

                    results.append({
                        "symbol": symbol,
                        "method": method,
                        "gia_T0": round(gia_T0, 2),
                        "gia_T1": round(gia_T1, 2),
                        "dinh_gia": round(fair_scaled, 2) if fair_scaled is not None else None,
                        "tín_hiệu": signal,
                        "thực_tế": thuc_te,
                        "đúng": is_correct
                    })
                except:
                    # Trường hợp lỗi, giữ tín hiệu GIU
                    results.append({
                        "symbol": symbol,
                        "method": method,
                        "gia_T0": round(gia_T0, 2),
                        "gia_T1": round(gia_T1, 2),
                        "dinh_gia": None,
                        "tín_hiệu": "GIU",
                        "thực_tế": thuc_te,
                        "đúng": False
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

# ==== 4 TAB CHÍNH ====
tab1, tab2, tab3, tab4 = st.tabs([
    "📌 Định giá riêng lẻ",
    "📈 Backtest",
    "🏆 Định giá Tổng hợp",
    "📊 Lọc cổ phiếu tiềm năng"
])

# ===== TAB 3: Định giá tổng hợp toàn bộ thị trường =====
with tab3:
    st.subheader("🏆 Định giá tổng hợp toàn bộ thị trường")

    # Hàm chuẩn hóa VND
    def to_vnd(x):
        try:
            v = float(x)
        except:
            return None
        if pd.isna(v):
            return None
        return v * 1000 if v < 1000 else v

    # Hàm lấy giá đóng cửa gần nhất (TCBS)
    def get_latest_market_price(symbol):
        today = datetime.now().date()
        weekday = today.weekday()
        if weekday == 5:
            trading_date = today - timedelta(days=1)
        elif weekday == 6:
            trading_date = today - timedelta(days=2)
        else:
            trading_date = today
        date_str = trading_date.strftime("%Y-%m-%d")

        try:
            quote = Quote(symbol=symbol, source="TCBS")
            df_history = quote.history(start=date_str, end=date_str)
            if df_history is not None and not df_history.empty:
                df_history['time'] = df_history['time'].astype(str)
                df_day = df_history[df_history['time'].str.startswith(date_str)]
                if not df_day.empty:
                    closing_price = float(df_day.iloc[-1]["close"])
                    return to_vnd(closing_price)
        except SystemExit as e:
            msg = str(e)
            if "rate limit" in msg.lower():
                match = re.search(r"sau (\d+) giây", msg)
                wait_seconds = int(match.group(1)) if match else 30
                st.warning(f"⚠️ Rate limit khi lấy giá {symbol}, chờ {wait_seconds} giây...")
                time.sleep(wait_seconds + 1)
                return get_latest_market_price(symbol)
        except Exception as e:
            st.write(f"⚠️ Không lấy được giá cho {symbol}: {e}")

        return None

    # Chọn chế độ trọng số
    use_auto_weight = st.checkbox("🎯 Sử dụng trọng số tự động theo ngành", value=True)

    ts_manual = {}
    if not use_auto_weight:
        st.markdown("### ⚖️ Nhập trọng số thủ công (áp dụng cho toàn bộ thị trường):")
        cols = st.columns(len(TRONG_SO_MAC_DINH))
        for i, key in enumerate(TRONG_SO_MAC_DINH):
            with cols[i]:
                ts_manual[key] = st.number_input(
                    f"{key}", min_value=0.0, max_value=1.0,
                    value=TRONG_SO_MAC_DINH[key], step=0.05, key=f"tab3_{key}"
                )

    if st.button("📊 Thực hiện định giá toàn bộ"):
        results = []
        try:
            all_sheets = load_data()
            for symbol, df in all_sheets.items():
                try:
                    if "period" not in df.columns or df["period"].dropna().empty:
                        continue
                    if not any(period_pattern.match(str(x)) for x in df["period"].dropna().astype(str).unique()):
                        continue

                    # lấy kỳ gần nhất
                    df["__pnum"] = df["period"].apply(period_to_sort_value)
                    latest = df.loc[df["__pnum"].idxmax()]

                    # dữ liệu đầu vào (chuẩn hóa như Tab 1)
                    eps  = get_value(latest, ["EPS", "earning_per_share"])
                    pe   = get_value(latest, ["P/E", "price_to_earning"])
                    pb   = get_value(latest, ["P/B", "price_to_book"])
                    roe  = get_value(latest, ["ROE", "return_on_equity"])
                    bvps = get_value(latest, ["BVPS", "book_value_per_share", "giá trị sổ sách", "giá trị sổ sách/cổ phiếu"])

                    fair = dinh_gia(eps, pe, pb, bvps, roe)

                    # chọn trọng số
                    if use_auto_weight:
                        industry = nganh_map.get(symbol, None) if 'nganh_map' in globals() else None
                        if industry and industry in TRONG_SO_THEO_NGANH:
                            ts_use = TRONG_SO_THEO_NGANH[industry]
                        else:
                            ts_use = TRONG_SO_MAC_DINH
                    else:
                        ts_use = ts_manual

                    weighted_methods = [k for k in fair.keys() if fair.get(k) is not None and k in ts_use]
                    ts_sum = sum(ts_use[k] for k in weighted_methods)
                    if ts_sum <= 0:
                        continue

                    gia_dinh_gia_vnd = sum(fair[k] * ts_use[k] for k in weighted_methods) / ts_sum

                    # giá thị trường
                    market_price_vnd = get_latest_market_price(symbol)
                    if market_price_vnd is None:
                        continue

                    # chênh lệch
                    chenh_vnd = gia_dinh_gia_vnd - market_price_vnd
                    ty_le = (chenh_vnd / market_price_vnd) * 100

                    if ty_le > 10:
                        khuyen_nghi = "MUA"
                    elif ty_le < -10:
                        khuyen_nghi = "BÁN"
                    else:
                        khuyen_nghi = "GIỮ"

                    results.append({
                        "Mã cổ phiếu": symbol,
                        "Định giá (trọng số)": float(gia_dinh_gia_vnd),
                        "Giá thị trường": float(market_price_vnd),
                        "Chênh lệch (VND)": float(chenh_vnd),
                        "% chênh lệch": float(round(ty_le, 2)),
                        "Khuyến nghị": khuyen_nghi
                    })
                except Exception as e:
                    st.write(f"⚠️ Lỗi khi xử lý {symbol}: {e}")
                    continue

            if results:
                df_top = pd.DataFrame(results)

                # format bảng
                df_view = df_top.copy()
                fmt = lambda x: f"{int(round(x, 0)):,}"
                df_view["Định giá (trọng số)"] = df_view["Định giá (trọng số)"].apply(fmt)
                df_view["Giá thị trường"]     = df_view["Giá thị trường"].apply(fmt)
                df_view["% chênh lệch"] = df_top.apply(
                    lambda r: f"{fmt(r['Chênh lệch (VND)'])} ({r['% chênh lệch']:.2f}%)", axis=1
                )
                df_view = df_view.drop(columns=["Chênh lệch (VND)"])

                st.dataframe(df_view, use_container_width=True)

                # xuất excel
                with io.BytesIO() as output:
                    with pd.ExcelWriter(output, engine="openpyxl") as writer:
                        df_top.to_excel(writer, index=False, sheet_name="Dinh_gia_all")
                    st.download_button("📥 Tải kết quả", data=output.getvalue(),
                                       file_name="dinh_gia_all.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.warning("⚠️ Không có kết quả nào.")
        except Exception as e:
            st.error(f"❌ Lỗi tổng: {e}")


# ===== TAB 4: Lọc cổ phiếu tiềm năng =====
with tab4:
    st.subheader("📊 Lọc cổ phiếu tiềm năng")

    DATA_FILE = "du_lieu_chung_khoan.xlsx"
    FILTER_FILE = "bo_loc.json"
    TEMP_FILE = "temp_results.xlsx"

    # --- Load file dữ liệu ---
    try:
        xls = pd.ExcelFile(DATA_FILE)
        all_symbols = xls.sheet_names
    except Exception as e:
        st.error(f"❌ Không đọc được file {DATA_FILE}: {e}")
        st.stop()

    # Lấy cột numeric mẫu từ 1 sheet
    sample_df = pd.read_excel(DATA_FILE, sheet_name=all_symbols[0])
    numeric_cols = [c for c in sample_df.columns if pd.api.types.is_numeric_dtype(sample_df[c])]

    # Gợi ý đơn vị hiển thị cho một số tiêu chí phổ biến
    UNIT_HINTS = {
        "EPS": "VNĐ/cp",
        "P/E": "lần",
        "P/B": "lần",
        "ROE": "%",
        "ROA": "%",
        "BVPS": "VNĐ/cp",
        "Doanh thu": "tỷ VNĐ",
        "% Tăng trưởng doanh thu": "%",
        "Lợi nhuận sau thuế": "tỷ VNĐ",
        "% Tăng trưởng lợi nhuận sau thuế": "%",
        "Tăng trưởng EPS": "%"
    }

    # --- Bộ lọc mẫu ---
    SAMPLE_FILTERS = {
        "Tăng trưởng ổn định": {
            "EPS": (">", 1000),
            "Tăng trưởng EPS": (">", 10),
            "Doanh thu": (">", 500),
            "% Tăng trưởng doanh thu": (">", 10),
            "Lợi nhuận sau thuế": (">", 50),
            "% Tăng trưởng lợi nhuận sau thuế": (">", 10),
        },
        "Hiệu quả sinh lời cao": {
            "ROE": (">", 15),
            "ROA": (">", 7),
            "EPS": (">", 2000),
            "Doanh thu": (">", 1000),
            "Lợi nhuận sau thuế": (">", 100),
        },
        "Định giá hấp dẫn (Value)": {
            "P/E": ("<", 12),
            "P/B": ("<", 1.5),
            "ROE": (">", 10),
            "EPS": (">", 1000),
        },
        "GARP – Tăng trưởng hợp lý": {
            "Tăng trưởng EPS": (">", 15),
            "ROE": (">", 12),
            "P/E": ("<", 15),
            "P/B": ("<", 2),
        },
        "Bluechip phòng thủ": {
            "Doanh thu": (">", 2000),
            "Lợi nhuận sau thuế": (">", 200),
            "ROE": (">", 10),
            "P/E": ("<", 18),
        }
    }

    st.markdown("### 📌 Bộ lọc mẫu")
    chosen_sample = st.selectbox("Chọn bộ lọc mẫu", ["-- Chọn --"] + list(SAMPLE_FILTERS.keys()))

    filter_state = {}
    if chosen_sample != "-- Chọn --":
        filter_state = SAMPLE_FILTERS[chosen_sample]
        st.success(f"✅ Đã load bộ lọc mẫu: {chosen_sample}")

    # --- Form nhập bộ lọc ---
    st.markdown("### ⚙️ Bộ lọc tiêu chí (có thể chỉnh sửa)")
    with st.form("filter_form"):
        for col in numeric_cols:
            col_left, col_mid, col_right = st.columns([2, 1, 2])
            with col_left:
                st.write(f"**{col}** ({UNIT_HINTS.get(col, '')})")
            with col_mid:
                op = st.selectbox("", [">", "<", "="],
                                  key=f"op_{col}",
                                  index={">": 0, "<": 1, "=": 2}.get(filter_state.get(col, (">", None))[0], 0))
            with col_right:
                val = st.number_input("", step=1, format="%d",
                                      key=f"val_{col}",
                                      value=int(filter_state.get(col, (">", 0))[1] or 0))
            if val is not None:
                filter_state[col] = (op, val)

        submitted = st.form_submit_button("💾 Cập nhật bộ lọc (chưa lọc)")
        if submitted:
            with open(FILTER_FILE, "w", encoding="utf-8") as f:
                json.dump(filter_state, f, ensure_ascii=False, indent=2)
            st.success("✅ Bộ lọc đã được lưu tạm thời")

    # --- Quản lý bộ lọc ---
    st.markdown("### 📂 Quản lý bộ lọc")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📂 Tải bộ lọc"):
            if os.path.exists(FILTER_FILE):
                with open(FILTER_FILE, "r", encoding="utf-8") as f:
                    filter_state = json.load(f)
                st.info("📂 Đã tải bộ lọc từ file")
            else:
                st.warning("⚠️ Chưa có file bộ lọc")
    with col2:
        if st.button("🗑️ Xóa bộ lọc"):
            if os.path.exists(FILTER_FILE):
                os.remove(FILTER_FILE)
                st.info("🗑️ Đã xóa bộ lọc")
            else:
                st.warning("⚠️ Không có file bộ lọc để xóa")
    with col3:
        if st.button("💾 Lưu bộ lọc vĩnh viễn"):
            if filter_state:
                with open(FILTER_FILE, "w", encoding="utf-8") as f:
                    json.dump(filter_state, f, ensure_ascii=False, indent=2)
                st.success("✅ Bộ lọc đã lưu thành file JSON")
            else:
                st.warning("⚠️ Chưa có tiêu chí để lưu")

    # --- Chọn periods ---
    st.markdown("### ⏱️ Chọn kỳ (period)")
    available_periods = sample_df["period"].dropna().unique().tolist() if "period" in sample_df.columns else []
    selected_periods = st.multiselect("Chọn kỳ muốn lọc", available_periods, default=available_periods[:1])

    # --- Chế độ lọc ---
    mode = st.radio("Chế độ lọc", ["Lọc mới", "Lọc tiếp"], horizontal=True)

    # --- Thực hiện lọc ---
    if st.button("🚀 Thực hiện lọc"):
        results = []
        logs = []

        if mode == "Lọc tiếp" and os.path.exists(TEMP_FILE):
            df_prev = pd.read_excel(TEMP_FILE)
            symbols_to_check = df_prev["Mã cổ phiếu"].tolist()
        else:
            symbols_to_check = all_symbols

        for symbol in symbols_to_check:
            try:
                df = pd.read_excel(DATA_FILE, sheet_name=symbol)
                if "period" not in df.columns:
                    continue
                df = df[df["period"].isin(selected_periods)]
                if df.empty:
                    continue

                df["__pnum"] = df["period"].apply(period_to_sort_value)
                latest = df.loc[df["__pnum"].idxmax()]

                ok = True
                reasons = []
                for col, (op, val) in filter_state.items():
                    cell = latest.get(col, None)
                    if pd.isna(cell):
                        ok = False; reasons.append(f"{col} thiếu dữ liệu")
                        continue
                    try:
                        cell_val = float(cell)
                        if op == ">" and not (cell_val > val):
                            ok = False; reasons.append(f"{col}={cell_val} <= {val}")
                        elif op == "<" and not (cell_val < val):
                            ok = False; reasons.append(f"{col}={cell_val} >= {val}")
                        elif op == "=" and not (cell_val == val):
                            ok = False; reasons.append(f"{col}={cell_val} ≠ {val}")
                    except Exception as e:
                        ok = False; reasons.append(f"Lỗi {col}: {e}")

                if ok:
                    results.append({"Mã cổ phiếu": symbol, "Kết quả": "ĐẠT"})
                    logs.append(f"{symbol}: ✅ ĐẠT")
                else:
                    results.append({"Mã cổ phiếu": symbol, "Kết quả": "Không đạt", "Lý do": "; ".join(reasons)})
                    logs.append(f"{symbol}: ❌ Không đạt ({'; '.join(reasons)})")

            except Exception as e:
                results.append({"Mã cổ phiếu": symbol, "Kết quả": "Lỗi", "Lý do": str(e)})
                logs.append(f"{symbol}: ⚠️ Lỗi ({e})")

        if results:
            df_res = pd.DataFrame(results)
            st.dataframe(df_res, use_container_width=True)

            # Hiển thị log chi tiết
            st.markdown("### 📜 Nhật ký lọc chi tiết")
            st.text_area("Log chi tiết", "\n".join(logs), height=300)

            # Lưu tạm để lọc tiếp
            with pd.ExcelWriter(TEMP_FILE, engine="openpyxl") as writer:
                df_res.to_excel(writer, sheet_name="Results", index=False)

            # Xuất Excel
            with io.BytesIO() as output:
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    df_res.to_excel(writer, sheet_name="Results", index=False)
                st.download_button("📥 Tải kết quả", data=output.getvalue(),
                                   file_name="ket_qua_loc.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.warning("⚠️ Không có kết quả nào")


# ===== TAB 1: Định giá riêng lẻ =====
with tab1:

    st.subheader("📌 Định giá cổ phiếu)")

    # --- Nhập mã ---
    symbol = st.text_input("Nhập mã cổ phiếu", value="FPT").upper()

    # --- Hàm xử lý chuỗi ---
    def remove_accents(input_str):
        nfkd_form = unicodedata.normalize('NFKD', str(input_str))
        return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

    # --- Load ngành & sàn từ file ---
    nganh_map, exchange_map = {}, {}
    try:
        ds = pd.read_csv("danh_sach_ma.csv", encoding="utf-8-sig")
        if {"symbol", "ngành", "exchange"}.issubset(ds.columns):
            nganh_map = dict(zip(ds["symbol"].astype(str).str.upper(), ds["ngành"]))
            exchange_map = dict(zip(ds["symbol"].astype(str).str.upper(), ds["exchange"]))
    except:
        try:
            ds = pd.read_csv("danh_sach_ma.csv", encoding="cp1258")
            if {"symbol", "ngành", "exchange"}.issubset(ds.columns):
                nganh_map = dict(zip(ds["symbol"].astype(str).str.upper(), ds["ngành"]))
                exchange_map = dict(zip(ds["symbol"].astype(str).str.upper(), ds["exchange"]))
        except:
            pass

    raw_industry = nganh_map.get(symbol, "Khác")
    exchange = exchange_map.get(symbol, "HOSE")

    # --- Mapping ngành ---
    INDUSTRY_KEYWORDS = {
        "Ngân hàng": ["bank", "ngan hang", "commercial bank", "banks"],
        "Chứng khoán": ["secur", "chung khoan", "broker", "brokerage", "ctck", "financial services"],
        "Bất động sản": ["real estate", "bat dong san", "property", "developer", "bất động sản"],
        "Xây dựng – VLXD": ["construct", "xay dung", "vlxd", "building", "construction", "construction & materials"],
        "Công nghệ – Viễn thông": ["tech", "cong nghe", "vien thong", "telecom", "it services"],
        "Bán lẻ – Tiêu dùng": ["retail", "ban le", "tieu dung", "consumer", "commerce", "bán lẻ"],
        "Thép – VL cơ bản": ["steel", "thep", "vl co ban", "basic resource", "iron", "basic resources"],
        "Điện – Năng lượng": ["power", "electric", "dien", "nang luong", "energy", "utilities", "điện"],
        "Dầu khí": ["oil", "gas", "dau", "khi", "petro", "petroleum", "xăng dầu", "khí đốt"],
        "Thực phẩm & Đồ uống": ["food", "drink", "do uong", "thuc pham", "thực phẩm", "beverage"],
        "Hóa chất": ["chemical", "hoa chat", "hóa chất"],
        "Truyền thông": ["media", "truyen thong"],
        "Hàng hóa công nghiệp": ["industrial goods", "industrial services", "goods & services"],
        "Hàng tiêu dùng": ["personal & household", "household goods"],
    }

    def map_industry(raw_industry):
        if not raw_industry or str(raw_industry).lower() == "nan":
            return "Khác"
        raw_norm = remove_accents(str(raw_industry)).lower()
        for vn_name, keywords in INDUSTRY_KEYWORDS.items():
            for kw in keywords:
                if kw in raw_norm:
                    return vn_name
        return "Khác"

    nganh = map_industry(raw_industry)
    st.info(f"📂 Ngành xác định: **{nganh}** | 🏛️ Sàn: **{exchange}**")

    # --- Trọng số mặc định theo ngành ---
    NGANH_TRONG_SO = {
        "Ngân hàng": {"P/E": 0.2, "P/B": 0.5, "ROE": 0.3, "PEG": 0.0, "DCF": 0.0},
        "Chứng khoán": {"P/E": 0.4, "P/B": 0.3, "ROE": 0.2, "PEG": 0.1, "DCF": 0.0},
        "Bất động sản": {"P/E": 0.2, "P/B": 0.4, "ROE": 0.0, "PEG": 0.0, "DCF": 0.4},
        "Xây dựng – VLXD": {"P/E": 0.4, "P/B": 0.0, "ROE": 0.2, "PEG": 0.0, "DCF": 0.4},
        "Công nghệ – Viễn thông": {"P/E": 0.3, "P/B": 0.0, "ROE": 0.1, "PEG": 0.4, "DCF": 0.2},
        "Bán lẻ – Tiêu dùng": {"P/E": 0.4, "P/B": 0.1, "ROE": 0.2, "PEG": 0.3, "DCF": 0.0},
        "Thép – VL cơ bản": {"P/E": 0.4, "P/B": 0.0, "ROE": 0.2, "PEG": 0.0, "DCF": 0.4},
        "Điện – Năng lượng": {"P/E": 0.3, "P/B": 0.2, "ROE": 0.0, "PEG": 0.0, "DCF": 0.5},
        "Dầu khí": {"P/E": 0.3, "P/B": 0.0, "ROE": 0.2, "PEG": 0.0, "DCF": 0.5},
        "Thực phẩm & Đồ uống": {"P/E": 0.4, "P/B": 0.2, "ROE": 0.2, "PEG": 0.2, "DCF": 0.0},
        "Hóa chất": {"P/E": 0.3, "P/B": 0.3, "ROE": 0.2, "PEG": 0.0, "DCF": 0.2},
        "Truyền thông": {"P/E": 0.4, "P/B": 0.2, "ROE": 0.2, "PEG": 0.2, "DCF": 0.0},
        "Hàng hóa công nghiệp": {"P/E": 0.3, "P/B": 0.3, "ROE": 0.2, "PEG": 0.0, "DCF": 0.2},
        "Hàng tiêu dùng": {"P/E": 0.4, "P/B": 0.2, "ROE": 0.2, "PEG": 0.2, "DCF": 0.0},
        "Khác": TRONG_SO_MAC_DINH
    }
    ts_auto = NGANH_TRONG_SO.get(nganh, TRONG_SO_MAC_DINH)

    # --- Chỉnh trọng số ---
    st.markdown("### ⚖️ Trọng số (tự động theo ngành, có thể chỉnh lại):")
    cols = st.columns(5)
    trong_so = {}
    for i, key in enumerate(TRONG_SO_MAC_DINH):
        default_val = ts_auto.get(key, TRONG_SO_MAC_DINH[key])
        with cols[i]:
            trong_so[key] = st.number_input(
                f"{key}",
                min_value=0.0,
                max_value=1.0,
                value=default_val,
                step=0.05,
                key=f"ts_{key}"
            )

    # --- Thực hiện định giá ---
    if st.button("🔍 Thực hiện định giá"):
        try:
            df = pd.read_excel("du_lieu_chung_khoan.xlsx", sheet_name=symbol)
            latest = df.iloc[0].to_dict()

            pe = latest.get("P/E")
            pb = latest.get("P/B")
            roe = latest.get("ROE")
            eps = latest.get("EPS")
            bvps = latest.get("Giá trị sổ sách/cổ phiếu")
            growth = latest.get("Tăng trưởng EPS", 0.1)
            r = 0.15

            st.markdown("### 📑 Dữ liệu tài chính quý gần nhất")
            st.dataframe(df.head(1))

            ket_qua = {}
            details = []

            if eps and pe:
                val = eps * pe
                ket_qua["P/E"] = val
                details.append(["P/E", f"EPS ({eps}) × P/E ({pe})", f"{val:,.0f} VND"])

            if bvps and pb:
                val = bvps * pb
                ket_qua["P/B"] = val
                details.append(["P/B", f"BVPS ({bvps}) × P/B ({pb})", f"{val:,.0f} VND"])

            if bvps and roe:
                val = bvps * roe * 10
                ket_qua["ROE"] = val
                details.append(["ROE", f"BVPS ({bvps}) × ROE ({roe}) × 10", f"{val:,.0f} VND"])

            # --- input cho PEG target (bạn có thể để ở trên cùng của Tab1 UI nếu muốn) ---
            peg_target = st.number_input("PEG target (PE/growth) — mặc định 1.0", value=1.0, step=0.1, format="%.2f")

            # --- chuẩn hóa growth (tránh dữ liệu lưu ở % thay vì decimal) ---
            def _normalize_growth(g):
                try:
                    if g is None:
                        return None
                    g = float(g)
                    # nếu g lớn hơn 2 (ví dụ 12 hoặc 11.8), rất có thể là percent -> chia 100
                    if abs(g) > 2:
                        return g / 100.0
                    return g
                except:
                    return None

            growth_norm = _normalize_growth(growth)  # growth lấy từ Excel trước đó

            # --- PEG (sửa lại chuẩn) ---
            gia_peg = None
            growth_norm = _normalize_growth(growth)  # chuẩn hóa growth: nếu là 12 thì thành 0.12
            if eps is not None and growth_norm is not None and growth_norm > 0:
                # PEG hợp lý = 1 => P/E hợp lý = Growth%
                fair_pe = growth_norm * 100  # growth decimal -> %
                gia_peg = float(eps) * fair_pe
                ket_qua["PEG"] = gia_peg
                details.append([
                    "PEG",
                    f"EPS({eps}) × Growth%({growth_norm*100:.2f})",
                    f"{gia_peg:,.0f} VND"
                ])
           
            if eps is not None and growth is not None and r > growth:
                try:
                    n = 5
                    # Chiết khấu EPS tăng trưởng 5 năm
                    dcf_val = 0
                    for i in range(1, n+1):
                        cf = eps * (1 + growth)**i
                        dcf_val += cf / (1 + r)**i

                    # Terminal Value tại cuối năm 5
                    cf5 = eps * (1 + growth)**n
                    tv = cf5 * (1 + growth) / (r - growth)
                    pv_tv = tv / (1 + r)**n

                    dcf_val += pv_tv
                    ket_qua["DCF"] = dcf_val

                    details.append([
                        "DCF (5 năm)", 
                        f"Σ EPS×(1+g)^i/(1+r)^i (i=1..5) + TV",
                        f"{dcf_val:,.0f} VND"
                    ])
                except Exception:
                    ket_qua["DCF"] = None

            # Tính giá trị hợp nhất
            valid_methods = {k: v for k, v in ket_qua.items() if v}
            tong_trong_so = sum(trong_so[k] for k in valid_methods.keys())
            gia_tri = sum(trong_so[k] * v for k, v in valid_methods.items()) / tong_trong_so if tong_trong_so > 0 else None

                    
            # ----- LẤY GIÁ THỊ TRƯỜNG (robust fallback) -----
            market_price = None
            price_source_note = None
            try:
                # try Quote realtime (VCI)
                quote = Quote(symbol=symbol, source="VCI")
                realtime_data = quote.realtime()
                # thử một số key phổ biến
                for key in ("priceMatched", "matchedPrice", "last", "price"):
                    if isinstance(realtime_data.get(key), (int, float, str)) and realtime_data.get(key) not in [None, ""]:
                        try:
                            raw = float(realtime_data.get(key))
                            market_price = raw * 1000 if raw < 1000 else raw
                            price_source_note = f"realtime ({key}) via Quote(VCI)"
                            break
                        except:
                            continue
            except Exception:
                market_price = None

            if market_price is None:
                # try Quote.history
                try:
                    quote = Quote(symbol=symbol, source="VCI")
                    today = datetime.now().date()
                    start = (today - timedelta(days=14)).strftime("%Y-%m-%d")
                    end = today.strftime("%Y-%m-%d")
                    df_price = quote.history(start=start, end=end)
                    if df_price is not None and not df_price.empty:
                        if "close" in df_price.columns:
                            last_close = df_price["close"].dropna().iloc[-1]
                            market_price = float(last_close) * 1000 if float(last_close) < 1000 else float(last_close)
                            price_source_note = "history via Quote(VCI)"
                except Exception:
                    market_price = None

            if market_price is None:
                # try reading GIA files as fallback (if they exist and have symbol)
                try:
                    if os.path.exists(GIA1_FILE):
                        g1 = pd.read_excel(GIA1_FILE)
                        if "symbol" in g1.columns and "close_price" in g1.columns:
                            row = g1[g1["symbol"].astype(str).str.upper() == symbol.upper()]
                            if not row.empty:
                                market_price = float(row.iloc[0]["close_price"])
                                price_source_note = f"from {GIA1_FILE}"
                    if market_price is None and os.path.exists(GIA0_FILE):
                        g0 = pd.read_excel(GIA0_FILE)
                        if "symbol" in g0.columns and "close_price" in g0.columns:
                            row = g0[g0["symbol"].astype(str).str.upper() == symbol.upper()]
                            if not row.empty:
                                market_price = float(row.iloc[0]["close_price"])
                                price_source_note = f"from {GIA0_FILE}"
                except Exception:
                    market_price = None

            # Hiển thị kết quả
            st.markdown("### 📊 Kết quả định giá")
            st.table(pd.DataFrame(details, columns=["Phương pháp", "Công thức", "Giá trị ước tính"]))

            if gia_tri:
                st.success(f"💰 Giá trị hợp nhất (theo trọng số): {gia_tri:,.0f} VND")

            if market_price:
                st.info(f"📈 Giá thị trường hiện tại: {market_price:,.0f} VND")
                chenh_lech = gia_tri - market_price
                ty_le = chenh_lech / market_price * 100
                st.info(f"🧮 Chênh lệch: **{chenh_lech:,.0f} VND** ({ty_le:.2f}%)")
                if gia_tri and gia_tri > market_price * 1.2:
                    st.success("✅ Khuyến nghị: MUA")
                elif gia_tri and gia_tri < market_price * 0.8:
                    st.error("❌ Khuyến nghị: BÁN")
                else:
                    st.warning("⚖️ Khuyến nghị: GIỮ")

        except Exception as e:
            st.error(f"Lỗi khi định giá: {e}")

# ===== TAB 2: BACKTEST =====
with tab2:
    st.subheader("📊 Backtest định giá (theo quý)")

    # Lấy danh sách period từ file du_lieu_chung_khoan
    bctc_all = load_data()
    periods = get_available_periods(bctc_all)
    selected_period = st.selectbox("Chọn quý để backtest:", periods, index=len(periods)-1)

    # File T0/T1
    t0_filename = f"T0_{selected_period}.xlsx"
    t1_filename = f"T1_{selected_period}.xlsx"

    if not os.path.exists(t0_filename) or not os.path.exists(t1_filename):
        st.error(f"Thiếu file {t0_filename} hoặc {t1_filename} trong thư mục.")
        st.stop()

    df_t0 = pd.read_excel(t0_filename)
    df_t1 = pd.read_excel(t1_filename)

    def _prepare_price_df(df):
        dfc = df.copy()
        sym_col = [c for c in dfc.columns if c.lower() in ("symbol","ticker","code")]
        price_col = [c for c in dfc.columns if "close" in c.lower() or "price" in c.lower()]
        if not sym_col or not price_col:
            return None
        dfc["symbol"] = dfc[sym_col[0]].astype(str).str.upper()
        dfc = dfc[["symbol", price_col[0]]].rename(columns={price_col[0]:"price"})
        return dfc

    df_t0 = _prepare_price_df(df_t0)
    df_t1 = _prepare_price_df(df_t1)

    if df_t0 is None or df_t1 is None:
        st.error("File T0/T1 thiếu cột symbol hoặc price.")
        st.stop()

    prices_T0 = dict(zip(df_t0["symbol"], df_t0["price"]))
    prices_T1 = dict(zip(df_t1["symbol"], df_t1["price"]))

    if st.button("▶ Chạy Backtest"):
        results = []
        for symbol, df in bctc_all.items():
            sym = str(symbol).upper()
            if sym not in prices_T0 or sym not in prices_T1:
                continue

            T0 = prices_T0[sym] * 1000
            T1 = prices_T1[sym] * 1000

            # Lấy đúng row dữ liệu BCTC của kỳ đang backtest
            df2 = df.copy()
            df2["period"] = df2["period"].astype(str)
            row = df2[df2["period"] == selected_period]
            if row.empty:
                continue
            row = row.iloc[0]

            eps = get_value(row, ["EPS"])
            pe = get_value(row, ["P/E"])
            pb = get_value(row, ["P/B"])
            roe = get_value(row, ["ROE"])
            bvps = get_value(row, ["BVPS", "Giá trị sổ sách/cổ phiếu"])
            growth = get_value(row, ["Tăng trưởng EPS"])

            # 🚨 Sử dụng công thức định giá chuẩn như Tab 1
            params = {"dcf_g": growth, "dcf_r": 0.15, "dcf_n": 5, "peg_target": 1.0}
            fair_dict = dinh_gia(eps, pe, pb, bvps, roe, params=params)

            # Thực tế: T1 > T0 hay T1 < T0
            thuc_te = "TANG" if T1 > T0 else ("GIAM" if T1 < T0 else "KHONG")

            fair_prices = dinh_gia(eps, pe, pb, bvps, roe)
            for method, fair in fair_dict.items():
                if fair is None:
                    continue
                signal = "MUA" if fair > T0 else "BAN"
                is_correct = (signal=="MUA" and T1>T0) or (signal=="BAN" and T1<T0)

                results.append({
                    "symbol": sym,
                    "period": selected_period,
                    "method": method,
                    "T0": round(T0,2),
                    "T1": round(T1,2),
                    "Fair": round(fair,2),
                    "Signal": signal,
                    "Actual": thuc_te,
                    "Correct": is_correct
                })

        if results:
            df_res = pd.DataFrame(results)
            st.write("### Kết quả chi tiết")
            st.dataframe(df_res)

            # Tổng hợp
            summary = df_res.groupby("method")["Correct"].agg(["count","sum"]).reset_index()
            summary["Accuracy (%)"] = summary["sum"] / summary["count"] * 100
            st.write("### Tổng hợp")
            st.dataframe(summary)

            # Xuất Excel
            out_file = f"ket_qua_backtest_{selected_period}.xlsx"
            with pd.ExcelWriter(out_file) as writer:
                df_res.to_excel(writer, sheet_name="ChiTiet", index=False)
                summary.to_excel(writer, sheet_name="TongHop", index=False)

            with open(out_file,"rb") as f:
                st.download_button(
                    "📥 Tải kết quả Excel",
                    f,
                    file_name=out_file,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.warning("Không có kết quả backtest.")


