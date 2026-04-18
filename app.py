import streamlit as st
import pandas as pd
import numpy as np
import onnxruntime as ort
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ==========================================
# --- 1. 加载 ONNX 模型 (AI 大脑) ---
# 作用：把我们在云端训练好的深度强化学习模型加载进网页内存。
# @st.cache_resource 确保模型只加载一次，防止每次点按钮都重新加载导致卡顿。
# ==========================================
@st.cache_resource
def load_model():
    return ort.InferenceSession("pump_d3qn_model.onnx")


try:
    session = load_model()
    input_name = session.get_inputs()[0].name
    model_loaded = True
except Exception as e:
    st.error(f"模型加载失败，请检查 onnx 文件是否存在。错误信息: {e}")
    model_loaded = False


# ==========================================
# --- 2. 专家导航仪 (安全护栏) ---
# 作用：复刻 MATLAB 中的传统水务专家经验逻辑。
# 根据当前时间、水位和电价，给出一个“推荐流量”，防止 AI 做出过于离谱的动作导致水池抽干或溢流。
# ==========================================
def get_expert_suggestion(target_hour, current_water_level, price, demand):
    base_price = 0.65
    valley_ratio = 0.48
    if price <= base_price * valley_ratio * 1.3:
        hours_left = max(7 - target_hour, 1) if target_hour <= 6 else 1
        dynamic_target = current_water_level + (3.95 - current_water_level) / hours_left
        k_gain = 3000
    elif 7 <= target_hour <= 12:
        progress = (target_hour - 6) / 6
        dynamic_target = 3.9 - (0.9 * progress)
        k_gain = 2000
    else:
        hours_left = 25 - target_hour
        dynamic_target = min(2.0 + (hours_left * 0.15), 3.9)
        k_gain = 1500

    level_gap = dynamic_target - current_water_level
    ideal_flow = max(demand + (level_gap * k_gain), 0)
    return ideal_flow, ideal_flow / 10000.0


# ==========================================
# --- 3. 多项式物理引擎 (真实环境仿真) ---
# 作用：1:1 像素级复刻 MATLAB 中的物理计算环境。
# 包含极其精确的三台水泵 Q-H(流量-扬程) 和 Q-P(流量-功率) 多项式系数。
# 用于计算在 AI 指定的某套开关动作下，真实能抽多少水、耗多少度电。
# ==========================================
def calculate_physics(action_real, current_ideal_flow):
    status = [0, 0, 0]
    if action_real == 1:
        status = [1, 0, 0]
    elif action_real == 2:
        status = [0, 1, 0]
    elif action_real == 3:
        status = [1, 1, 0]
    elif action_real == 4:
        status = [0, 0, 1]
    elif action_real == 5:
        status = [1, 0, 1]
    elif action_real == 6:
        status = [0, 1, 1]
    elif action_real == 7:
        status = [1, 1, 1]
    else:
        status = [1, 0, 0]

    design_head = 15.0
    pump_h = [[-1e-6, 0.0008, 27.001], [-8e-7, 0.0002, 28.000], [-8e-7, 0.0005, 27.096]]
    pump_p = [[1e-6, 0.0142, 127.83], [3e-7, 0.0195, 129.55], [-3e-6, 0.0337, 132.93]]

    cal_rated_q = [0.0, 0.0, 0.0]
    cal_rated_p = [0.0, 0.0, 0.0]

    for i in range(3):
        c_eff = pump_h[i][2] - design_head
        delta = pump_h[i][1] ** 2 - 4 * pump_h[i][0] * c_eff
        if delta >= 0:
            cal_rated_q[i] = max((-pump_h[i][1] - np.sqrt(delta)) / (2 * pump_h[i][0]), 0)
        else:
            cal_rated_q[i] = 0
        cal_rated_p[i] = pump_p[i][0] * (cal_rated_q[i] ** 2) + pump_p[i][1] * cal_rated_q[i] + pump_p[i][2]

    current_max_flow = sum([cal_rated_q[i] * status[i] for i in range(3)])
    real_total_flow = min(current_ideal_flow, current_max_flow)
    real_total_power = 0.0

    if current_max_flow > 0:
        load_ratio = real_total_flow / current_max_flow
        for i in range(3):
            if status[i] == 1:
                real_total_power += cal_rated_p[i] * (0.4 + 0.6 * load_ratio)

    return real_total_flow, real_total_power, status


# ==========================================
# --- 4. 基础数据配置 (电价与网页框架) ---
# ==========================================
normal_prices = [0.312] * 6 + [0.65] * 6 + [0.312] * 2 + [0.65] * 2 + [0.9685] * 2 + [1.17] * 2 + [0.9685] * 4
summer_prices = [0.312] * 6 + [0.65] * 6 + [0.312] * 2 + [0.65] * 2 + [0.9685] * 4 + [1.17] * 2 + [0.9685] * 2

st.set_page_config(page_title="泵房智能调度系统", layout="wide")
st.title("💧 给水厂取水泵房智能调度系统")
st.markdown("基于 Dueling DQN 深度强化学习的自动化泵组启停规划平台")

with st.sidebar:
    st.header("⚙️ 初始环境设置")
    init_level = st.slider("跨日初始水位 (m)", min_value=1.8, max_value=4.0, value=2.5, step=0.1)
    init_action = st.selectbox("起始泵组状态", options=[1, 2, 4],
                               format_func=lambda x: {1: "单开1号泵", 2: "单开2号泵", 4: "单开3号泵"}[x])

tab1, tab2 = st.tabs(["⏱️ 单日精细调度监控", "📅 长周期宏观评估 (多日/全年)"])

# ==========================================
# --- 选项卡 1：单日精细调度 (用于查看一天的微观动作) ---
# ==========================================
with tab1:
    st.subheader("📊 1. 输入今日管网特征与需水量预测")
    uploaded_file = st.file_uploader("📂 快捷操作：上传单日需水量表格", type=["xlsx", "csv"], key="upload_day")

    default_demand = pd.DataFrame({
        "时间": [f"{i}:00" for i in range(1, 25)],
        "需水量 (m³)": [1500, 1400, 1300, 1300, 1400, 1800, 2500, 3200, 3000, 2800, 2700, 2600, 2500, 2400, 2400, 2500,
                        2800, 3100, 3300, 3100, 2600, 2200, 1800, 1600],
        "实时电价 (元)": normal_prices
    })

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)
            if "需水量" in df_upload.columns:
                default_demand["需水量 (m³)"] = df_upload["需水量"].values[:24]
            elif "需水量 (m³)" in df_upload.columns:
                default_demand["需水量 (m³)"] = df_upload["需水量 (m³)"].values[:24]
            else:
                default_demand["需水量 (m³)"] = df_upload.iloc[:, 0].values[:24]
            st.success(f"✅ 读取成功！")
        except Exception as e:
            st.error(f"❌ 读取失败。详情: {e}")

    user_input_df = st.data_editor(default_demand, use_container_width=True, hide_index=True, key="editor_day")

    # 单日推演按钮
    if st.button("🚀 生成单日最优方案", type="primary", disabled=not model_loaded, key="btn_day"):
        current_level = init_level
        last_action = init_action
        pump_run_hours = [0, 0, 0]
        schedule_results = []
        total_daily_cost = 0.0
        progress_bar = st.progress(0)

        for hour in range(1, 25):
            demand = user_input_df.loc[hour - 1, "需水量 (m³)"]
            price = user_input_df.loc[hour - 1, "实时电价 (元)"]

            ideal_flow, normalized_ideal_flow = get_expert_suggestion(hour, current_level, price, demand)
            state_vector = np.array([current_level, hour, price, last_action, demand / 1000.0,
                                     min(pump_run_hours[0], 15), min(pump_run_hours[1], 15), min(pump_run_hours[2], 15),
                                     normalized_ideal_flow], dtype=np.float32)

            q_values = session.run(None, {input_name: state_vector.reshape(1, -1)})[0]
            action_real = np.argmax(q_values) + 1
            real_flow, real_power, status = calculate_physics(action_real, ideal_flow)

            hourly_cost = real_power * price
            total_daily_cost += hourly_cost

            for i in range(3):
                if status[i] == 1:
                    pump_run_hours[i] += 1
                else:
                    pump_run_hours[i] = 0

            level_drop = (demand - real_flow) / 2992.5
            new_level = max(1.8, min(current_level - level_drop, 4.0))

            action_dict = {1: "1号泵", 2: "2号泵", 3: "1+2号泵", 4: "3号泵", 5: "1+3号泵", 6: "2+3号泵", 7: "三泵全开"}
            schedule_results.append({"时刻": f"{hour}:00", "需水量 (m³)": demand, "电价": price,
                                     "AI 决策": action_dict.get(action_real, "异常"),
                                     "实际抽水 (m³)": round(real_flow, 1), "耗电功率 (kW)": round(real_power, 1),
                                     "电费 (元)": round(hourly_cost, 2), "水位 (m)": round(new_level, 3)})

            current_level = new_level
            last_action = action_real
            progress_bar.progress(hour / 24.0)

        st.success(f"✅ 调度推演完成！今日预计总电费：**{round(total_daily_cost, 2)} 元**")
        results_df = pd.DataFrame(schedule_results)
        st.dataframe(results_df, use_container_width=True)

        st.subheader("📈 预测水位与分时电价交互图")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=results_df["时刻"], y=results_df["电价"], name="分时电价", mode="lines", line_shape="vh",
                       fill="tozeroy", fillcolor="rgba(46, 204, 113, 0.15)",
                       line=dict(color="rgba(46, 204, 113, 0.6)", width=2),
                       hovertemplate="<b>电价:</b> %{y:.3f} 元<extra></extra>"), secondary_y=True)
        fig.add_trace(go.Scatter(x=results_df["时刻"], y=results_df["水位 (m)"], name="预测水位", mode="lines+markers",
                                 line=dict(color="#3498db", width=3, shape="spline"),
                                 marker=dict(size=8, color="#2980b9", symbol="circle"),
                                 customdata=results_df["AI 决策"],
                                 hovertemplate="<b>时间:</b> %{x}<br><b>水位:</b> %{y:.2f} m<br><span style='color:#e74c3c'><b>AI 动作: %{customdata}</b></span><extra></extra>"),
                      secondary_y=False)
        fig.add_hline(y=1.8, line_dash="dash", line_color="#e74c3c", opacity=0.8, annotation_text="防汽蚀红线",
                      secondary_y=False)
        fig.add_hline(y=4.0, line_dash="dash", line_color="#c0392b", opacity=0.8, annotation_text="物理溢流线",
                      secondary_y=False)
        fig.update_layout(height=400, hovermode="x unified",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_yaxes(title_text="水池水位 (m)", color="#2980b9", range=[1.5, 4.2], secondary_y=False)
        fig.update_yaxes(title_text="实时电价 (元)", color="#27ae60", range=[0, 1.5], secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("⏱️ 水泵机组 24 小时排班甘特图")
        pump_data = []
        for i, row in results_df.iterrows():
            hour_end = int(row["时刻"].split(":")[0])
            hour_start = hour_end - 1
            decision = row["AI 决策"]
            if "1号" in decision or "三泵全开" in decision: pump_data.append(
                {"Pump": "1号泵", "Start": hour_start, "Duration": 1})
            if "2号" in decision or "三泵全开" in decision: pump_data.append(
                {"Pump": "2号泵", "Start": hour_start, "Duration": 1})
            if "3号" in decision or "三泵全开" in decision: pump_data.append(
                {"Pump": "3号泵", "Start": hour_start, "Duration": 1})

        df_gantt = pd.DataFrame(pump_data)
        fig_gantt = go.Figure()
        colors = {"1号泵": "rgba(231, 76, 60, 0.9)", "2号泵": "rgba(243, 156, 18, 0.9)",
                  "3号泵": "rgba(46, 204, 113, 0.9)"}
        if not df_gantt.empty:
            for pump in ["3号泵", "2号泵", "1号泵"]:
                pump_df = df_gantt[df_gantt["Pump"] == pump]
                if not pump_df.empty:
                    fig_gantt.add_trace(
                        go.Bar(base=pump_df["Start"], x=pump_df["Duration"], y=pump_df["Pump"], orientation='h',
                               marker_color=colors[pump], name=pump,
                               hovertemplate="<b>%{y}</b><br>运行时间段: %{base}:00 至 %{customdata}:00<extra></extra>",
                               customdata=pump_df["Start"] + 1))
        fig_gantt.update_layout(height=300, barmode='overlay', margin=dict(l=20, r=20, t=40, b=20),
                                plot_bgcolor="rgba(248, 249, 250, 1)",
                                xaxis=dict(title="<b>时刻 (0:00 - 24:00)</b>", tickmode='linear', tick0=0, dtick=1,
                                           range=[0, 24], gridcolor='rgba(200, 200, 200, 0.2)'),
                                yaxis=dict(title="", gridcolor='rgba(200, 200, 200, 0.2)'), showlegend=False)
        st.plotly_chart(fig_gantt, use_container_width=True)

# ==========================================
# --- 选项卡 2：长周期宏观评估 (用于 39 天或全年的宏观验证) ---
# ==========================================
with tab2:
    st.subheader("📅 多日/全年自动仿真引擎")
    st.markdown(
        "上传连续多日的 **需水量数据矩阵 (24行 × N列)**，系统将自动拼接长周期序列，并动态套用时令电价进行全闭环无缝推演。")

    uploaded_long_file = st.file_uploader("📂 上传长周期需水矩阵表 (.xlsx)", type=["xlsx", "csv"], key="upload_long")

    # 外层结构：判断文件是否上传成功
    if uploaded_long_file is not None:
        try:
            # 1. 文件读取与智能清洗 (去除文字表头)
            if uploaded_long_file.name.endswith('.csv'):
                df_long = pd.read_csv(uploaded_long_file, header=None)
            else:
                df_long = pd.read_excel(uploaded_long_file, header=None)

            df_numeric = df_long.apply(pd.to_numeric, errors='coerce')
            df_numeric = df_numeric.dropna(axis=0, how='all').dropna(axis=1, how='all')

            # 2. 判断数据是否有效 (行数必须 >= 24)
            if df_numeric.shape[0] >= 24:
                matrix_24xN = df_numeric.iloc[-24:, :].values
                demand_sequence = matrix_24xN.flatten('F')
                total_hours = len(demand_sequence)
                total_days = total_hours // 24

                st.info(f"✅ 解析成功！检测到有效矩阵特征，共计 **{total_days} 天 ({total_hours} 小时)**，已准备就绪。")

                # ----------------------------------------------------
                # A. 核心运算区：只有点击按钮那一刻才会执行
                # ----------------------------------------------------
                if st.button("🚀 启动长周期全自动推演", type="primary", disabled=not model_loaded, key="btn_long"):
                    current_level = init_level
                    last_action = init_action
                    pump_run_hours = [0, 0, 0]
                    total_cost = 0.0
                    total_water = 0.0
                    long_results = []

                    start_date = pd.to_datetime("2026-01-01")
                    progress_bar_long = st.progress(0)

                    # 开启物理步进大循环 (比如 8760 个小时)
                    for h_idx in range(total_hours):
                        day_idx = h_idx // 24
                        hour_of_day = (h_idx % 24) + 1
                        current_date = start_date + pd.Timedelta(days=day_idx)

                        is_summer = current_date.month in [7, 8]
                        price = summer_prices[hour_of_day - 1] if is_summer else normal_prices[hour_of_day - 1]
                        demand = demand_sequence[h_idx]

                        ideal_flow, normalized_ideal_flow = get_expert_suggestion(hour_of_day, current_level, price,
                                                                                  demand)
                        state_vector = np.array([current_level, hour_of_day, price, last_action, demand / 1000.0,
                                                 min(pump_run_hours[0], 15), min(pump_run_hours[1], 15),
                                                 min(pump_run_hours[2], 15), normalized_ideal_flow], dtype=np.float32)

                        q_values = session.run(None, {input_name: state_vector.reshape(1, -1)})[0]
                        action_real = np.argmax(q_values) + 1

                        real_flow, real_power, status = calculate_physics(action_real, ideal_flow)
                        hourly_cost = real_power * price
                        total_cost += hourly_cost
                        total_water += real_flow

                        for i in range(3):
                            if status[i] == 1:
                                pump_run_hours[i] += 1
                            else:
                                pump_run_hours[i] = 0

                        level_drop = (demand - real_flow) / 2992.5
                        current_level = max(1.8, min(current_level - level_drop, 4.0))
                        last_action = action_real

                        action_dict = {1: "1号泵", 2: "2号泵", 3: "1+2号泵", 4: "3号泵", 5: "1+3号泵", 6: "2+3号泵",
                                       7: "三泵全开"}
                        long_results.append({
                            "日期": current_date.strftime("%Y-%m-%d"),
                            "时刻": f"{hour_of_day}:00",
                            "需水量(m³)": round(demand, 1),
                            "实际抽水(m³)": round(real_flow, 1),
                            "水泵状态": action_dict.get(action_real, "异常"),
                            "总功率(kW)": round(real_power, 1),
                            "电价(元)": price,
                            "电费(元)": round(hourly_cost, 2),
                            "水位(m)": round(current_level, 3)
                        })

                        if h_idx % 100 == 0 or h_idx == total_hours - 1:
                            progress_bar_long.progress((h_idx + 1) / total_hours)

                    # 运算完毕，把数据封存在“网页记忆”中，防止点击下拉框时数据丢失
                    st.session_state['long_run_done'] = True
                    st.session_state['df_long_results'] = pd.DataFrame(long_results)
                    st.session_state['total_days'] = total_days
                    st.session_state['total_water'] = total_water
                    st.session_state['total_cost'] = total_cost
                    st.session_state['total_demand'] = sum(demand_sequence)
                    st.session_state['total_hours'] = total_hours

                # ----------------------------------------------------
                # B. 数据展示区：只要记忆里有数据，就保持渲染显示
                # （此段代码与上面的 if st.button 完美平齐）
                # ----------------------------------------------------
                if st.session_state.get('long_run_done', False):
                    # 1. 从记忆中唤醒数据
                    df_long_results = st.session_state['df_long_results']
                    total_days = st.session_state['total_days']
                    total_water = st.session_state['total_water']
                    total_cost = st.session_state['total_cost']
                    total_demand = st.session_state['total_demand']
                    total_hours = st.session_state['total_hours']

                    # 2. 渲染顶部 5 大核心指标 (KPI)
                    st.success("✅ 推演完毕！长周期运行宏观报告如下：")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("仿真总跨度", f"{total_days} 天")
                    col2.metric("累计总需水量", f"{total_demand / 10000:.2f} 万 m³")
                    col3.metric("累计总供水量", f"{total_water / 10000:.2f} 万 m³")
                    col4.metric("总电费成本", f"{total_cost / 10000:.2f} 万元")
                    unit_cost = (total_cost / total_water * 1000) if total_water > 0 else 0
                    col5.metric("千吨水调度成本", f"{unit_cost:.2f} 元/千吨")

                    # 3. 数据预处理与绘制宏观月度图
                    df_long_results["日期"] = pd.to_datetime(df_long_results["日期"])
                    df_long_results["月份"] = df_long_results["日期"].dt.strftime("%Y-%m")

                    st.markdown("#### 📈 长周期宏观趋势分析 (月度聚合)")
                    df_monthly = df_long_results.groupby("月份").agg(
                        {"需水量(m³)": "sum", "电费(元)": "sum"}).reset_index()

                    fig_month = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_month.add_trace(go.Bar(x=df_monthly["月份"], y=df_monthly["电费(元)"], name="月度总电费",
                                               marker_color="rgba(243, 156, 18, 0.7)"), secondary_y=True)
                    fig_month.add_trace(go.Scatter(x=df_monthly["月份"], y=df_monthly["需水量(m³)"], name="月度总需水",
                                                   mode="lines+markers", line=dict(color="#3498db", width=3)),
                                        secondary_y=False)
                    fig_month.update_layout(height=400, hovermode="x unified",
                                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right",
                                                        x=1), margin=dict(l=20, r=20, t=40, b=20),
                                            plot_bgcolor="rgba(248, 249, 250, 1)")
                    fig_month.update_yaxes(title_text="<b>月度总需水量 (m³)</b>", secondary_y=False)
                    fig_month.update_yaxes(title_text="<b>月度总电费 (元)</b>", secondary_y=True)
                    st.plotly_chart(fig_month, use_container_width=True)

                    # 4. 交互式下拉框与微观日度细节图 (数据钻取)
                    st.markdown("#### 🔍 月度明细钻取 (日度聚合)")
                    available_months = df_monthly["月份"].tolist()
                    selected_month = st.selectbox("请选择要查看具体日期的月份：", options=available_months)

                    if selected_month:
                        df_selected = df_long_results[df_long_results["月份"] == selected_month]
                        df_daily = df_selected.groupby("日期").agg(
                            {"需水量(m³)": "sum", "电费(元)": "sum"}).reset_index()
                        df_daily["日期"] = df_daily["日期"].dt.strftime("%Y-%m-%d")

                        fig_day = make_subplots(specs=[[{"secondary_y": True}]])
                        fig_day.add_trace(
                            go.Bar(x=df_daily["日期"], y=df_daily["电费(元)"], name=f"{selected_month} 每日电费",
                                   marker_color="rgba(46, 204, 113, 0.6)"), secondary_y=True)
                        fig_day.add_trace(
                            go.Scatter(x=df_daily["日期"], y=df_daily["需水量(m³)"], name=f"{selected_month} 每日需水",
                                       mode="lines+markers", line=dict(color="#2980b9", width=2)), secondary_y=False)
                        fig_day.update_layout(height=350, hovermode="x unified", margin=dict(l=20, r=20, t=20, b=20),
                                              plot_bgcolor="rgba(248, 249, 250, 1)")
                        fig_day.update_yaxes(title_text="<b>每日需水量 (m³)</b>", secondary_y=False)
                        fig_day.update_yaxes(title_text="<b>每日电费 (元)</b>", secondary_y=True)
                        st.plotly_chart(fig_day, use_container_width=True)

                    # 5. 底部的“一键报表导出”按钮 (永远显示在最后)
                    st.markdown("---")
                    st.write(f"💡 系统已自动将所有物理引擎产生的 {total_hours} 条精细状态数据装订成册。")
                    csv_data = df_long_results.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(label="📥 一键下载完整详细调度报表 (.csv)", data=csv_data,
                                       file_name=f"综合调度推演报告_{total_days}天.csv", mime="text/csv",
                                       type="primary")

            # 应对上面 if df_numeric.shape[0] >= 24 的异常处理
            else:
                st.warning("⚠️ 格式无法识别：请确保 Excel 至少包含 24 行连续的需水量数据。")

        # 应对上面 try 文件读取失败的异常捕获
        except Exception as e:
            st.error(f"❌ 数据解析失败。详情: {e}")