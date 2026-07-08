import streamlit as st
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
import os

# --- 1. 網頁標題與設定 ---
st.set_page_config(page_title="F121 節能與製程預測系統", layout="wide")
st.title("🔥 F121 天然氣最佳化操作與 C122 溫度預測系統")

# --- 2. 智慧讀取資料與訓練模型 ---
@st.cache_resource
def train_models_with_real_data():
    # 智慧偵測：不管你上傳的是 .xlsx 還是 .csv，只要有名稱就通殺
    excel_file = "F121_Data.xlsx"
    csv_file = "F121_Data.csv"
    
    df = None
    
    # 優先嘗試讀取真正的 Excel 檔案
    if os.path.exists(excel_file):
        try:
            xl = pd.ExcelFile(excel_file)
            # 如果有多個分頁，自動找含有真實數據的第二個分頁(index=1)，找不到就拿第一個
            sheet_idx = 1 if len(xl.sheet_names) > 1 else 0
            df = pd.read_excel(excel_file, sheet_name=sheet_idx, skiprows=[1])
        except Exception:
            pass
            
    # 如果 Excel 讀不到，嘗試讀取 CSV 檔案
    if df is None and os.path.exists(csv_file):
        try:
            # 這裡用 openpyxl 強制嘗試，因為你的 CSV 本質可能是 Excel
            df = pd.read_excel(csv_file, sheet_name=1, skiprows=[1])
        except Exception:
            try:
                df = pd.read_excel(csv_file, sheet_name=0, skiprows=[1])
            except Exception:
                try:
                    df = pd.read_csv(csv_file, skiprows=[1], encoding='utf-8')
                except Exception:
                    df = pd.read_csv(csv_file, skiprows=[1], encoding='big5')

    if df is None:
        st.error("❌ 找不到有效的資料檔（F121_Data.xlsx 或 F121_Data.csv），請確認檔案已正確上傳至 GitHub。")
        st.stop()
        
    # 【超級清洗】移除欄位名稱中所有的換行、特殊空白、多餘空格
    df.columns = df.columns.astype(str).str.replace('\n', ' ').str.replace('\r', ' ')
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # 建立模糊搜尋函式，利用關鍵字自動鎖定正確的欄位
    def find_col(keywords, default_name):
        for col in df.columns:
            if any(k in col.lower() for k in keywords):
                return col
        return default_name

    dt_col = find_col(['dt', 'operation'], 'DT operation')
    c141_col = find_col(['c141', 'operation'], 'C141 operation')
    clo_col = find_col(['clo', 'circulation', 'flow'], 'F121 CLO circulation flow')
    temp_col = find_col(['f121outlet', 'temperature'], 'F121outlet temperature')
    oxy_col = find_col(['oxygen'], 'F121 Oxygen content %')
    ng_col = find_col(['ng', 'consumption'], 'F121 NG consumption')
    c122_col = find_col(['c122', 'bottom'], 'C122 bottom temperature')

    X_cols = [dt_col, c141_col, clo_col, temp_col, oxy_col]
    all_cols = X_cols + [ng_col, c122_col]
    
    # 將數據強制轉換為數字並剔除文字行與空值
    df_clean = df[all_cols].apply(pd.to_numeric, errors='coerce').dropna()
    
    if len(df_clean) == 0:
        st.error("❌ 找不到流水帳原始數據！請確認您的 Excel 檔中是否包含完整的『每日歷史數據流水帳』。")
        st.stop()

    X = df_clean[X_cols]
    y_ng = df_clean[ng_col]
    y_c122 = df_clean[c122_col]
    
    # 獲取各變數的真實上下限範圍
    bounds_dict = {
        'dt': (float(X[dt_col].min()), float(X[dt_col].max())),
        'c141': (float(X[c141_col].min()), float(X[c141_col].max())),
        'clo': (float(X[clo_col].min()), float(X[clo_col].max())),
        'temp': (float(X[temp_col].min()), float(X[temp_col].max())),
        'oxy': (float(X[oxy_col].min()), float(X[oxy_col].max()))
    }
    
    # 訓練模型
    model_ng = LGBMRegressor(random_state=42)
    model_ng.fit(X, y_ng)
    
    model_c122 = LGBMRegressor(random_state=42)
    model_c122.fit(X, y_c122)
    
    return model_ng, model_c122, bounds_dict

with st.spinner("🚀 正在自動智慧辨識檔案格式並訓練 AI 模型..."):
    model_ng, model_c122, bounds = train_models_with_real_data()

# --- 3. 側邊欄：不可控變數輸入 ---
st.sidebar.header("📋 當前不可控排程設定")
input_dt = st.sidebar.slider("DT operation (稼動率)", min_value=bounds['dt'][0], max_value=bounds['dt'][1], value=(bounds['dt'][0]+bounds['dt'][1])/2, step=0.01)
input_c141 = st.sidebar.slider("C141 operation (稼動率)", min_value=bounds['c141'][0], max_value=bounds['c141'][1], value=(bounds['c141'][0]+bounds['c141'][1])/2, step=0.01)

# --- 4. 優化演算法核心 ---
def objective_func(controllable_vars):
    features = np.array([[input_dt, input_c141, controllable_vars[0], controllable_vars[1], controllable_vars[2]]])
    return model_ng.predict(features)[0]

opt_bounds = [bounds['clo'], bounds['temp'], bounds['oxy']]
initial_guess = [(opt_bounds[i][0] + opt_bounds[i][1])/2 for i in range(3)]

res = minimize(objective_func, initial_guess, bounds=opt_bounds, method='SLSQP')
best_flow, best_temp, best_oxy = res.x[0], res.x[1], res.x[2]

# --- 5. 預測最佳操作下的 C122 溫度 ---
best_features = np.array([[input_dt, input_c141, best_flow, best_temp, best_oxy]])
predicted_c122_temp = model_c122.predict(best_features)[0]

# --- 6. 主要內容區：顯示最佳化與預測結果 ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("💡 系統推薦最佳操作參數")
    st.metric(label="🔹 F121 CLO circulation flow (最佳流量)", value=f"{best_flow:.3f}")
    st.metric(label="🔹 F121outlet temperature (最佳出口溫度)", value=f"{best_temp:.2f} °C")
    st.metric(label="🔹 F121 Oxygen content % (最佳含氧量)", value=f"{best_oxy:.2f} %")

with col2:
    st.subheader("📊 預估效益與製程監控")
    st.info(f"✨ 在目前的排程下，預估最低天然氣消耗量 Y 為： **{res.fun:.2f}**")
    st.success(f"🌡️ 此最佳操作狀態下，預估的 **C122 bottom temperature** 為： **{predicted_c122_temp:.2f} °C**")

# --- 7. 互動式測試：手動微調 ---
st.markdown("---")
st.subheader("🎮 手動操作與即時溫度/能耗連動模擬器")

c_flow = st.slider("手動調整 CLO flow", bounds['clo'][0], bounds['clo'][1], float(best_flow))
c_temp = st.slider("手動調整 出口溫度", bounds['temp'][0], bounds['temp'][1], float(best_temp))
c_oxy = st.slider("手動調整 含氧量 %", bounds['oxy'][0], bounds['oxy'][1], float(best_oxy))

manual_features = np.array([[input_dt, input_c141, c_flow, c_temp, c_oxy]])
manual_y = model_ng.predict(manual_features)[0]
manual_c122 = model_c122.predict(manual_features)[0]

res_col1, res_col2 = st.columns(2)
res_col1.metric(label="🏃 手動設定下的預估天然氣消耗 (Y)", value=f"{manual_y:.2f}")
res_col2.metric(label="🌡️ 手動設定下的預估 C122 塔底溫度", value=f"{manual_c122:.2f} °C")
