import streamlit as st
import pandas as pd
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 解决图表中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# --- 1. 加载 ONNX 模型 ---
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


# --- 2. 1:1 复刻 MATLAB 的专家导航仪 ---
def get_expert_suggestion(target_hour, current_water_level, price, demand):
    base_price = 0.65
    valley_ratio = 0.48

    # 策略 A：谷电时间感知蓄水模式
    if price <= base_price * valley_ratio * 1.3:
        hours_left = max(7 - target_hour, 1) if target_hour <= 6 else 1
        dynamic_target = current_water_level + (3.95 - current_water_level) / hours_left
        k_gain = 3000
    # 策略 B：早高峰平滑释水模式
    elif 7 <= target_hour <= 12:
        progress = (target_hour - 6) / 6
        dynamic_target = 3.9 - (0.9 * progress)
        k_gain = 2000
    # 策略 C：常规时段稳态过渡模式
    else:
        hours_left = 25 - target_hour
        dynamic_target = min(2.0 + (hours_left * 0.15), 3.9)
        k_gain = 1500

    level_gap = dynamic_target - current_water_level
    ideal_flow = max(demand + (level_gap * k_gain), 0)
    normalized_flow = ideal_flow / 10000.0

    return ideal_flow, normalized_flow


# --- 3. 1:1 复刻 MATLAB 的多项式物理引擎 (含 Q-H, Q-P, VFD变频) ---
def calculate_physics(action_real, current_ideal_flow):
    # 动作解码为状态掩码
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
        status = [1, 0, 0]  # 兜底保护

    design_head = 15.0
    # 精确对齐你代码里的 Q-H 参数
    pump_h = [
        [-1e-6, 0.0008, 27.001],
        [-8e-7, 0.0002, 28.000],
        [-8e-7, 0.0005, 27.096]
    ]
    # 精确对齐你代码里的 Q-P 参数
    pump_p = [
        [1e-6, 0.0142, 127.83],
        [3e-7, 0.0195, 129.55],
        [-3e-6, 0.0337, 132.93]
    ]

    cal_rated_q = [0.0, 0.0, 0.0]
    cal_rated_p = [0.0, 0.0, 0.0]

    # 计算额定流量和额定功率
    for i in range(3):
        c_eff = pump_h[i][2] - design_head
        delta = pump_h[i][1] ** 2 - 4 * pump_h[i][0] * c_eff
        if delta >= 0:
            q_val = (-pump_h[i][1] - np.sqrt(delta)) / (2 * pump_h[i][0])
            cal_rated_q[i] = max(q_val, 0)
        else:
            cal_rated_q[i] = 0

        cal_rated_p[i] = pump_p[i][0] * (cal_rated_q[i] ** 2) + pump_p[i][1] * cal_rated_q[i] + pump_p[i][2]

    # 变频控制系统 (VFD) 模拟
    current_max_flow = sum([cal_rated_q[i] * status[i] for i in range(3)])
    real_total_flow = min(current_ideal_flow, current_max_flow)
    real_total_power = 0.0

    if current_max_flow > 0:
        load_ratio = real_total_flow / current_max_flow
        for i in range(3):
            if status[i] == 1:
                # 严格使用你的变频耗电公式：0.4 + 0.6 * LoadRatio
                p_val = cal_rated_p[i] * (0.4 + 0.6 * load_ratio)
                real_total_power += p_val

    return real_total_flow, real_total_power, status


# --- 4. 网页前端布局 ---
st.set_page_config(page_title="泵房智能调度系统", layout="wide")
st.title("💧 给水厂取水泵房智能调度系统")
st.markdown("基于 Dueling DQN 深度强化学习的 24 小时全自动泵组启停规划")

with st.sidebar:
    st.header("⚙️ 初始环境设置")
    init_level = st.slider("初始水位 (m)", min_value=1.8, max_value=4.0, value=2.5, step=0.1)
    init_action = st.selectbox("昨日 24:00 泵组状态", options=[1, 2, 4], format_func=lambda x: {1:"单开1号泵", 2:"单开2号泵", 4:"单开3号泵"}[x])

st.subheader("📊 1. 输入今日 24 小时管网特征参数")

# 默认电价与需水量模板
normal_prices = [0.312]*6 + [0.65]*6 + [0.312]*2 + [0.65]*2 + [0.9685]*2 + [1.17]*2 + [0.9685]*4
default_demand = pd.DataFrame({
    "时间": [f"{i}:00" for i in range(1, 25)],
    "需水量 (m³)": [1500, 1400, 1300, 1300, 1400, 1800, 2500, 3200, 3000, 2800, 2700, 2600, 2500, 2400, 2400, 2500, 2800, 3100, 3300, 3100, 2600, 2200, 1800, 1600],
    "实时电价 (元)": normal_prices
})

# 🌟 新增：文件上传功能
uploaded_file = st.file_uploader("📂 快捷操作：上传外部需水量预测表格 (支持 .xlsx 或 .csv)", type=["xlsx", "csv"])

if uploaded_file is not None:
    try:
        # 判断并读取上传的文件
        if uploaded_file.name.endswith('.csv'):
            df_upload = pd.read_csv(uploaded_file)
        else:
            df_upload = pd.read_excel(uploaded_file)

        # 智能匹配：寻找叫 "需水量" 或 "需水量 (m³)" 的列
        if "需水量" in df_upload.columns:
            default_demand["需水量 (m³)"] = df_upload["需水量"].values[:24]
        elif "需水量 (m³)" in df_upload.columns:
            default_demand["需水量 (m³)"] = df_upload["需水量 (m³)"].values[:24]
        else:
            # 如果找不到特定的表头，就默认强行抓取第一列的前24个数据
            default_demand["需水量 (m³)"] = df_upload.iloc[:, 0].values[:24]

        st.success(f"✅ 成功读取文件 `{uploaded_file.name}`！数据已自动填入下方表格，您还可以进行手动微调。")
    except Exception as e:
        st.error(f"❌ 读取文件失败，请确保表格包含连续的 24 行数字。错误详情: {e}")

# 显示可编辑表格 (如果用户上传了文件，这里显示的就是上传后的数据)
user_input_df = st.data_editor(default_demand, use_container_width=True, hide_index=True)

# --- 5. 核心推理推演 ---
if st.button("🚀 一键生成最优调度方案", type="primary", disabled=not model_loaded):
    current_level = init_level
    last_action = init_action
    pump_run_hours = [0, 0, 0]

    schedule_results = []
    total_daily_cost = 0.0  # 记录全天总电费

    progress_bar = st.progress(0)

    for hour in range(1, 25):
        demand = user_input_df.loc[hour - 1, "需水量 (m³)"]
        price = user_input_df.loc[hour - 1, "实时电价 (元)"]

        # 1. 呼叫专家导航仪
        ideal_flow, normalized_ideal_flow = get_expert_suggestion(hour, current_level, price, demand)

        # 2. 构造 9 维状态
        state_vector = np.array([
            current_level,
            hour,
            price,
            last_action,
            demand / 1000.0,
            min(pump_run_hours[0], 15),
            min(pump_run_hours[1], 15),
            min(pump_run_hours[2], 15),
            normalized_ideal_flow
        ], dtype=np.float32)

        # 3. 大脑思考，输出动作
        q_values = session.run(None, {input_name: state_vector.reshape(1, -1)})[0]
        action_real = np.argmax(q_values) + 1

        # 4. 物理引擎接管：计算真实抽水量和功率
        real_flow, real_power, status = calculate_physics(action_real, ideal_flow)

        # 计算电费成本
        hourly_cost = real_power * price
        total_daily_cost += hourly_cost

        # 5. 更新疲劳时间
        for i in range(3):
            if status[i] == 1:
                pump_run_hours[i] += 1
            else:
                pump_run_hours[i] = 0

        # 6. 计算水位变化 (严格使用你的面积参数 2992.5)
        level_drop = (demand - real_flow) / 2992.5
        new_level = current_level - level_drop
        new_level = max(1.8, min(new_level, 4.0))

        # 7. 记录数据
        action_dict = {1: "1号泵", 2: "2号泵", 3: "1+2号泵", 4: "3号泵", 5: "1+3号泵", 6: "2+3号泵", 7: "三泵全开"}
        schedule_results.append({
            "时刻": f"{hour}:00",
            "需水量 (m³)": demand,
            "电价": price,
            "AI 决策": action_dict.get(action_real, "异常"),
            "实际抽水 (m³)": round(real_flow, 1),
            "耗电功率 (kW)": round(real_power, 1),
            "电费 (元)": round(hourly_cost, 2),
            "水位 (m)": round(new_level, 3)
        })

        # 8. 状态跨时段交接
        current_level = new_level
        last_action = action_real
        progress_bar.progress(hour / 24.0)

    st.success(f"✅ 调度推演完成！今日预计总电费：**{round(total_daily_cost, 2)} 元**")

    results_df = pd.DataFrame(schedule_results)
    st.dataframe(results_df, use_container_width=True)

    st.subheader("📈 预测水位与分时电价交互图 (支持鼠标悬停与缩放)")

    # 创建双 Y 轴图表
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 🌟 图层 1：电价背景 (阶梯面积图，放右轴)
    fig.add_trace(
        go.Scatter(
            x=results_df["时刻"],
            y=results_df["电价"],
            name="分时电价 (元)",
            mode="lines",
            line_shape="vh",  # 阶梯状折线
            fill="tozeroy",  # 填充至 X 轴
            fillcolor="rgba(46, 204, 113, 0.15)",  # 极淡的绿色背景
            line=dict(color="rgba(46, 204, 113, 0.6)", width=2),
            hovertemplate="<b>电价:</b> %{y:.3f} 元<extra></extra>"
        ),
        secondary_y=True,
    )

    # 🌟 图层 2：水位走势 (带圆点的折线图，放左轴)
    fig.add_trace(
        go.Scatter(
            x=results_df["时刻"],
            y=results_df["水位 (m)"],
            name="预测水位 (m)",
            mode="lines+markers",
            line=dict(color="#3498db", width=3, shape="spline"),  # 平滑曲线
            marker=dict(size=8, color="#2980b9", symbol="circle"),
            # 将 AI 的动作埋入悬停提示中！
            customdata=results_df["AI 决策"],
            hovertemplate=(
                    "<b>时间:</b> %{x}<br>" +
                    "<b>水位:</b> %{y:.2f} m<br>" +
                    "<span style='color:#e74c3c'><b>AI 动作: %{customdata}</b></span><extra></extra>"
            )
        ),
        secondary_y=False,
    )

    # 🌟 图层 3：物理红线 (水平警戒线)
    fig.add_hline(y=1.8, line_dash="dash", line_color="#e74c3c", opacity=0.8, annotation_text="防汽蚀红线 (1.8m)",
                  annotation_position="bottom right", secondary_y=False)
    fig.add_hline(y=4.0, line_dash="dash", line_color="#c0392b", opacity=0.8, annotation_text="物理溢流线 (4.0m)",
                  annotation_position="top right", secondary_y=False)

    # 全局排版优化
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",  # 统一悬停框，一目了然
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(248, 249, 250, 1)",  # 极淡的高级灰底色
    )

    # 设置坐标轴范围和样式
    fig.update_yaxes(title_text="<b>水池水位 (m)</b>", color="#2980b9", range=[1.5, 4.2], secondary_y=False)
    fig.update_yaxes(title_text="<b>实时电价 (元)</b>", color="#27ae60", range=[0, 1.5], secondary_y=True)

    # 将高保真图表渲染到网页
    st.plotly_chart(fig, use_container_width=True)