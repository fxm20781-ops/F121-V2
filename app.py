import streamlit as st
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
import os

# --- 1. 網頁標題與設定 ---
st.set_page_config(page_title="F121 節能與製程預測系統", layout="wide")
st.title("🔥 F121 天然氣最佳化操作與 C122 溫度預測系統")

# --- 2. 讀取真實資料與訓練雙模型 ---
@st.cache_resource
def train_models_with_real_data():
    csv_filename = "F121_Data.csv"
    
    if not os.path.exists(csv_filename):
        st.error(f"❌ 找不到資料檔 {csv_filename}，請確認是否有上傳到 GitHub。")
        st.stop()
        
    # 讀取 CSV：跳過第 1 行的 Tag 代號 (TR122-11 等)
    df = pd.read_csv(csv_filename, skiprows=[1]) 
    
    # 清理欄位名稱（移除換行符號與前後空格）
    df.columns = df.columns.astype(str).str.replace('\n', ' ').str.replace('\r', ' ')
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # 定義對應的真實欄位名稱
    X_cols = ['DT operation', 'C141 operation', 'F121 CLO circulation flow', 'F121outlet temperature', 'F121 Oxygen content %']
    y_ng_col = 'F121 NG consumption'
    y_c122_col = 'C122 bottom temperature'
    
    # 移除非數值或缺失值的資料
    all_cols = X_cols + [y_ng_col, y_c122_col]
    df_clean = df[all_cols].apply(pd.to_numeric, errors='coerce').dropna()
    
    X = df_clean[X_cols]
    y_ng = df_clean[y_ng_col]
    y_c122 = df_clean[y_c122_col]
    
    # 獲取各變數的真實上下限
    bounds_dict = {col: (float(X[col].min()), float(X[col].max())) for col in X_cols}
    
    # 訓練模型
    model_ng = LGBMRegressor(random_state=42)
    model_ng.fit(X, y_ng)
    
    model_c122 = LGBMRegressor(random_state=42)
    model_c122.fit(X, y_c122)
    
    return model_ng, model_c122, bounds_dict

with st.spinner("🚀 正在讀取歷史數據並訓練 AI 模型..."):
    model_ng, model_c122, bounds = train_models_with_real_data()

# --- 3. 側邊欄：不可控變數輸入 ---
st.sidebar.header("📋 當前不可控排程設定")
input_dt = st.sidebar.slider("DT operation (稼動率)", min_value=bounds['DT operation'][0], max_value=bounds['DT operation'][1], value=(bounds['DT operation'][0]+bounds['DT operation'][1])/2, step=0.01)
input_c141 = st.sidebar.slider("C141 operation (稼動率)", min_value=bounds['C141 operation'][0], max_value=bounds['C141 operation'][1], value=(bounds['C141 operation'][0]+bounds['C141 operation'][1])/2, step=0.01)

# --- 4. 優化演算法核心 ---
def objective_func(controllable_vars):
    features = np.array([[input_dt, input_c141, controllable_vars[0], controllable_vars[1], controllable_vars[2]]])
    return model_ng.predict(features)[0]

opt_bounds = [
    bounds['F121 CLO circulation flow'],
    bounds['F121outlet temperature'],
    bounds['F121 Oxygen content %']
]
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

c_flow = st.slider("手動調整 CLO flow", bounds['F121 CLO circulation flow'][0], bounds['F121 CLO circulation flow'][1], float(best_flow))
c_temp = st.slider("手動調整 出口溫度", bounds['F121outlet temperature'][0], bounds['F121outlet temperature'][1], float(best_temp))
c_oxy = st.slider("手動調整 含氧量 %", bounds['F121 Oxygen content %'][0], bounds['F121 Oxygen content %'][1], float(best_oxy))

manual_features = np.array([[input_dt, input_c141, c_flow, c_temp, c_oxy]])
manual_y = model_ng.predict(manual_features)[0]
manual_c122 = model_c122.predict(manual_features)[0]

res_col1, res_col2 = st.columns(2)
res_col1.metric(label="🏃 手動設定下的預估天然氣消耗 (Y)", value=f"{manual_y:.2f}")
res_col2.metric(label="🌡️ 手動設定下的預估 C122 塔底溫度", value=f"{manual_c122:.2f} °C")import streamlit as st
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
import os

# --- 1. 網頁標題與設定 ---
st.set_page_config(page_title="F121 節能與製程預測系統", layout="wide")
st.title("🔥 F121 天然氣最佳化操作與 C122 溫度預測系統")

# --- 2. 讀取真實資料與訓練雙模型 (自動快取) ---
@st.cache_resource
def train_models_with_real_data():
    csv_filename = "F121_Data.csv" # 👈 請確保這張表跟 app.py 放再同一個 GitHub 資料夾
    
    if not os.path.exists(csv_filename):
        st.error(f"❌ 找不到資料檔 {csv_filename}，請確認是否有上傳到 GitHub。")
        st.stop()
        
    # 讀取 CSV (跳過前兩行的單位/標籤資訊，從第0行開始讀，但清理欄位)
    df = pd.read_csv(csv_filename, skiprows=[1, 2]) 
    
    # 清理欄位名稱（移除換行符號與前後空格）
    df.columns = df.columns.str.replace('\n', ' ').str.strip()
    
    # 定義對應的真實欄位名稱
    # 根據您的檔案：'DT operation', 'C141 operation', 'C122 bottom temperature', 'F121 CLO circulation flow', 'F121outlet temperature', 'F121 NG consumption', 'F121 Oxygen content %'
    X_cols = ['DT operation', 'C141 operation', 'F121 CLO circulation flow', 'F121outlet temperature', 'F121 Oxygen content %']
    y_ng_col = 'F121 NG consumption'
    y_c122_col = 'C122 bottom temperature'
    
    # 移除非數值或缺失值的資料
    all_cols = X_cols + [y_ng_col, y_c122_col]
    df_clean = df[all_cols].dropna().apply(pd.to_numeric, errors='coerce').dropna()
    
    X = df_clean[X_cols]
    y_ng = df_clean[y_ng_col]
    y_c122 = df_clean[y_c122_col]
    
    # 獲取各變數的真實上下限，供後續滑桿與優化使用
    bounds_dict = {col: (float(X[col].min()), float(X[col].max())) for col in X_cols}
    
    # 訓練模型 1：預測天然氣
    model_ng = LGBMRegressor(random_state=42)
    model_ng.fit(X, y_ng)
    
    # 訓練模型 2：預測 C122 溫度
    model_c122 = LGBMRegressor(random_state=42)
    model_c122.fit(X, y_c122)
    
    return model_ng, model_c122, bounds_dict

with st.spinner("🚀 正在讀取歷史數據並訓練 AI 模型..."):
    model_ng, model_c122, bounds = train_models_with_real_data()

# --- 3. 側邊欄：不可控變數輸入 (依據歷史真實範圍) ---
st.sidebar.header("📋 當前不可控排程設定")
dt_min, dt_max = bounds['DT operation']
c141_min, c141_max = bounds['C141 operation']

input_dt = st.sidebar.slider("DT operation (稼動率)", min_value=dt_min, max_value=dt_max, value=(dt_min+dt_max)/2, step=0.01)
input_c141 = st.sidebar.slider("C141 operation (稼動率)", min_value=c141_min, max_value=c141_max, value=(c141_min+c141_max)/2, step=0.01)

# --- 4. 優化演算法核心（針對天然氣尋優） ---
def objective_func(controllable_vars):
    # controllable_vars = [CLO_flow, F121_temp, Oxygen]
    # 這裡確保傳入模型的是標準的 2D 矩陣格式 (1 rows, 5 columns)
    features = np.array([[input_dt, input_c141, controllable_vars[0], controllable_vars[1], controllable_vars[2]]])
    return model_ng.predict(features)[0]

# 設定可控變數尋優邊界 (來自真實歷史數據範圍)
opt_bounds = [
    bounds['F121 CLO circulation flow'],
    bounds['F121outlet temperature'],
    bounds['F121 Oxygen content %']
]
initial_guess = [(opt_bounds[i][0] + opt_bounds[i][1])/2 for i in range(3)]

# 執行優化
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

# --- 7. 互動式測試：手動微調與即時連動 ---
st.markdown("---")
st.subheader("🎮 手動操作與即時溫度/能耗連動模擬器")

flow_min, flow_max = bounds['F121 CLO circulation flow']
temp_min, temp_max = bounds['F121outlet temperature']
oxy_min, oxy_max = bounds['F121 Oxygen content %']

c_flow = st.slider("手動調整 CLO flow", flow_min, flow_max, float(best_flow))
c_temp = st.slider("手動調整 出口溫度", temp_min, temp_max, float(best_temp))
c_oxy = st.slider("手動調整 含氧量 %", oxy_min, oxy_max, float(best_oxy))

# 手動預測結果 (確保使用 2D 陣列防止報錯)
manual_features = np.array([[input_dt, input_c141, c_flow, c_temp, c_oxy]])
manual_y = model_ng.predict(manual_features)[0]
manual_c122 = model_c122.predict(manual_features)[0]

res_col1, res_col2 = st.columns(2)
res_col1.metric(label="🏃 手動設定下的預估天然氣消耗 (Y)", value=f"{manual_y:.2f}")
res_col2.metric(label="🌡️ 手動設定下的預估 C122 塔底溫度", value=f"{manual_c122:.2f} °C")
