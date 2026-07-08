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
    excel_file = "F121_Data.xlsx"
    csv_file = "F121_Data.csv"
    
    df = None
    
    if os.path.exists(excel_file):
        try:
            xl = pd.ExcelFile(excel_file)
            sheet_idx = 1 if len(xl.sheet_names) > 1 else 0
            df = pd.read_excel(excel_file, sheet_name=sheet_idx, skiprows=[1])
        except Exception:
            pass
            
    if df is None and os.path.exists(csv_file):
        try:
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
        st.error("❌ 找不到有效的資料檔，請確認 F121_Data.xlsx 有上傳到 GitHub。")
        st.stop()
        
    # 清理欄位名稱（移除換行、將多個空格壓扁成單一空格）
    df.columns = df.columns.astype(str).str.replace('\n', ' ').str.replace('\r', ' ')
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # 【超級核心修正】完全模糊比對欄位，不再死綁固定字串
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
    
    df_clean = df[all_cols].apply(pd.to_numeric, errors='coerce').dropna()
    
    if len(df_clean) == 0:
        st.error("❌ 數據解析後為空，請確認檔案內包含真實數據。")
        st.stop()

    X = df_clean[X_cols]
    y_ng = df_clean[ng_col]
    y_c122 = df_clean[c122_col]
    
    # 使用動態比對到的欄位名稱來建立邊界
    bounds_dict = {
        'dt': (float(X[dt_col].min()), float(X[dt_col].max())),
        'c141': (float(X[c141_col].min()), float(X[c141_col].max())),
        'clo': (float(X[clo_col].min()), float(X[clo_col].max())),
        'temp': (float(X[temp_col].min()), float(X[temp_col].max())),
        'oxy': (float(X[oxy_col].min()), float(X[oxy_col].max()))
    }
    
    model_ng = LGBMRegressor(random_state=42)
    model_ng.fit(X, y_ng)
    
    model_c122 = LGBMRegressor(random_state=42)
    model_c122.fit(X, y_c122)
    
    # 把動態確定的欄位名稱也回傳，供介面顯示
    col_names = {'dt': dt_col, 'c141': c141_col, 'clo': clo_col, 'temp': temp_col, 'oxy': oxy_col}
    return model_ng, model_c122, bounds_dict, col_names

with st.spinner("🚀 正在智慧辨識欄位並訓練 AI 模型..."):
    model_ng, model_c122, bounds, names = train_models_with_real_data()

# --- 3. 側邊欄：不可控變數輸入 ---
st.sidebar.header("📋 當前不可控排程設定")
input_dt = st.sidebar.slider(f"{names['dt']} (稼動率)", min_value=bounds['dt'][0], max_value=bounds['dt'][1], value=(bounds['dt'][0]+bounds['dt'][1])/2, step=0.01)
input_c141 = st.sidebar.slider(f"{names['c141']} (稼動率)", min_value=bounds['c141'][0], max_value=bounds['c141'][1], value=(bounds['c141'][0]+bounds['c141'][1])/2, step=0.01)

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
    st.metric(label=f"🔹 {names['clo']} (最佳流量)", value=f"{best_flow:.3f}")
    st.metric(label=f"🔹 {names['temp']} (最佳出口溫度)", value=f"{best_temp:.2f} °C")
    st.metric(label=f"🔹 {names['oxy']} (最佳含氧量)", value=f"{best_oxy:.2f} %")

with col2:
    st.subheader("📊 預估效益與製程監控")
    st.info(f"✨ 在目前的排程下，預估最低天然氣消耗量 Y 為： **{res.fun:.2f}**")
    st.success(f"🌡️ 此最佳操作狀態下，預估的 **C122 bottom temperature** 為： **{predicted_c122_temp:.2f} °C**")

# --- 7. 互動式測試：手動微調 ---
st.markdown("---")
st.subheader("🎮 手動操作與即時溫度/能耗連動模擬器")

c_flow = st.slider(f"手動調整 {names['clo']}", bounds['clo'][0], bounds['clo'][1], float(best_flow))
c_temp = st.slider(f"手動調整 {names['temp']}", bounds['temp'][0], bounds['temp'][1], float(best_temp))
c_oxy = st.slider(f"手動調整 {names['oxy']}", bounds['oxy'][0], bounds['oxy'][1], float(best_oxy))

manual_features = np.array([[input_dt, input_c141, c_flow, c_temp, c_oxy]])
manual_y = model_ng.predict(manual_features)[0]
manual_c122 = model_c122.predict(manual_features)[0]

res_col1, res_col2 = st.columns(2)
res_col1.metric(label="🏃 手動設定下的預估天然氣消耗 (Y)", value=f"{manual_y:.2f}")
res_col2.metric(label="🌡️ 手動設定下的預估 C122 塔底溫度", value=f"{manual_c122:.2f} °C")

