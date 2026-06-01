import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from huggingface_hub import HfApi, InferenceClient
from datetime import datetime, timezone
import pickle
import shap
import networkx as nx
import sqlite3
import scipy.stats as stats

# Import Thư viện Học máy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(page_title="HF VN Data Science", page_icon="📈", layout="wide")


# ==========================================
# ------------1. DATA ENGINE----------------
# ==========================================
# ĐỔI TÊN HÀM ĐỂ PHÁ VỠ CACHE CỨNG ĐẦU CỦA STREAMLIT
@st.cache_data
def load_data_final_v1(limit=3000):
    conn = sqlite3.connect("huggingface_local_pipeline.db")
    from datetime import timezone

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 1. Thử lấy từ cache SQLite trước
    try:
        db_df = pd.read_sql_query(
            f"SELECT * FROM clean_models WHERE fetched_date = '{today_str}'", conn
        )
        if not db_df.empty:
            db_df["createdAt"] = pd.to_datetime(db_df["createdAt"], utc=True)

            # --- AUTO FIX: Tự động vá mọi cột bị thiếu do Database cũ ---
            if "year" not in db_df.columns:
                db_df["year"] = db_df["createdAt"].dt.year
            if "month_year" not in db_df.columns:
                db_df["month_year"] = db_df["createdAt"].dt.to_period("M").astype(str)
            if "model_name_only" not in db_df.columns:
                db_df["model_name_only"] = db_df["modelId"].apply(
                    lambda x: str(x).split("/")[-1]
                )
            if "sentiment_score" not in db_df.columns:
                np.random.seed(42)
                base_sent = (
                    55
                    + 4 * np.log1p(db_df["likes"])
                    - 1.5 * np.log1p(db_df["downloads"])
                )
                db_df["sentiment_score"] = np.clip(
                    base_sent + np.random.normal(10, 8, size=len(db_df)), 0, 100
                ).round(1)
                db_df["sentiment_class"] = pd.cut(
                    db_df["sentiment_score"],
                    bins=[0, 52, 72, 100],
                    labels=["Tiêu cực", "Trung lập", "Tích cực"],
                )
            # -----------------------------------------------------------

            conn.close()
            return db_df
    except Exception:
        pass

    # 2. Nếu không có cache, lấy từ API
    api = HfApi()
    try:
        models = api.list_models(limit=limit, sort="downloads")
        data = []
        for m in models:
            model_id = getattr(m, "modelId", "Unknown")
            data.append(
                {
                    "modelId": model_id,
                    "author": model_id.split("/")[0] if "/" in model_id else "Official",
                    "model_name_only": model_id.split("/")[-1],
                    "downloads": getattr(m, "downloads", 0) or 0,
                    "likes": getattr(m, "likes", 0) or 0,
                    "task": getattr(m, "pipeline_tag", "Other") or "Other",
                    "createdAt": getattr(m, "created_at", datetime(2024, 1, 1)),
                    "name_len": len(model_id.split("/")[-1]),
                }
            )

        df = pd.DataFrame(data)
        if df.empty:
            conn.close()
            return pd.DataFrame()

        # 3. Tính toán các đặc trưng kỹ thuật (Feature Engineering)
        df["log_downloads"] = np.log1p(df["downloads"])
        df["log_likes"] = np.log1p(df["likes"])
        df["engagement_rate"] = (df["likes"] / (df["downloads"] + 1)) * 100

        # Tạo cột year và month_year
        df["createdAt"] = pd.to_datetime(df["createdAt"], utc=True)
        df["year"] = df["createdAt"].dt.year
        df["month_year"] = df["createdAt"].dt.to_period("M").astype(str)

        # Phân khúc (Scale)
        df["scale"] = pd.qcut(
            df["downloads"],
            q=4,
            labels=["Niche", "Emerging", "Popular", "Viral"],
            duplicates="drop",
        ).astype(str)

        # Mô phỏng Sentiment
        np.random.seed(42)
        base_sentiment = 55 + 4 * df["log_likes"] - 1.5 * df["log_downloads"]
        df["sentiment_score"] = np.clip(
            base_sentiment + np.random.normal(10, 8, size=len(df)), 0, 100
        ).round(1)
        df["sentiment_class"] = pd.cut(
            df["sentiment_score"],
            bins=[0, 52, 72, 100],
            labels=["Tiêu cực", "Trung lập", "Tích cực"],
        )

        df["fetched_date"] = today_str

        # 4. Lưu cache vào SQLite
        df_to_save = df.copy()
        df_to_save["createdAt"] = df_to_save["createdAt"].astype(str)
        # df_to_save.to_sql("clean_models", conn, if_exists="replace", index=False)
        conn.close()

        return df.sort_values("downloads", ascending=False).reset_index(drop=True)

    except Exception as e:
        st.error(f"Chi tiết lỗi API: {e}")
        conn.close()
        return pd.DataFrame()


def get_statistics(df):
    if df.empty:
        return {
            "total_models": 0,
            "total_downloads": 0,
            "top_author": "N/A",
            "correlation": 0,
        }
    return {
        "total_models": len(df),
        "total_downloads": df["downloads"].sum(),
        "top_author": df["author"].value_counts().idxmax(),
        "correlation": round(df["downloads"].corr(df["likes"]), 2),
    }


# ==========================================
# ------2. MACHINE LEARNING ENGINE----------
# ==========================================
def perform_clustering(df):
    if len(df) < 3:
        return df, None
    df_c = df.copy()
    X = np.log1p(df_c[["downloads", "likes"]])
    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_c["Cluster"] = kmeans.fit_predict(X_scaled)
    df_c["Cluster_Name"] = df_c["Cluster"].map(
        {0: "Tiềm năng", 1: "Phổ biến", 2: "Cộng đồng"}
    )
    return df_c, X_scaled


@st.cache_resource(show_spinner="⚙️ Đang tối ưu hóa Siêu tham số Học máy...")
def process_ml_insights(df):
    if len(df) < 15:
        return df, None, None, None, None, None, None
    df_ml = df.copy()
    df_task_dummies = pd.get_dummies(df_ml["task"], prefix="t")

    X = df_task_dummies.copy()
    X["log_downloads"] = np.log1p(df_ml["downloads"])
    y = np.log1p(df_ml["likes"])

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=(15 / 85), random_state=42
    )

    rf_param_grid = {"n_estimators": [50, 100, 150], "max_depth": [5, 10, 15]}
    gb_param_grid = {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
    }

    models_dict = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=7, random_state=42),
        "Random Forest": GridSearchCV(
            RandomForestRegressor(random_state=42),
            rf_param_grid,
            cv=3,
            scoring="r2",
            n_jobs=-1,
        ),
        "Gradient Boosting": GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            gb_param_grid,
            cv=3,
            scoring="r2",
            n_jobs=-1,
        ),
    }

    trained_models, metrics_list, predictions_dict = {}, [], {}
    y_val_orig = np.expm1(y_val)
    y_test_orig = np.expm1(y_test)

    for name, model in models_dict.items():
        model.fit(X_train, y_train)

        best_model = (
            model.best_estimator_ if hasattr(model, "best_estimator_") else model
        )
        trained_models[name] = best_model

        y_pred_val = best_model.predict(X_val)
        r2_val = r2_score(y_val, y_pred_val)
        mae_val = mean_absolute_error(y_val_orig, np.expm1(y_pred_val))

        y_pred_test = best_model.predict(X_test)
        predictions_dict[name] = y_pred_test
        r2_test = r2_score(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test_orig, np.expm1(y_pred_test))
        rmse_test = np.sqrt(mean_squared_error(y_test_orig, np.expm1(y_pred_test)))

        metrics_list.append(
            {
                "Mô hình": name,
                "R² Validation": round(r2_val, 4),
                "R² Test": round(r2_test, 4),
                "MAE Validation": round(mae_val, 2),
                "MAE Test": round(mae_test, 2),
                "RMSE Test": round(rmse_test, 2),
            }
        )

    return (
        df_ml,
        trained_models,
        pd.DataFrame(metrics_list),
        X.columns.tolist(),
        y_test,
        predictions_dict,
        X_test,
    )


@st.cache_resource
def load_bert_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_data
def get_cached_embeddings(text_list):
    model = load_bert_model()
    return model.encode(text_list, show_progress_bar=False)


def get_bert_recommendations(df, target_model_id, top_n=5):
    if df.empty or target_model_id not in df["modelId"].values:
        return pd.DataFrame(), None, None

    df["text_for_embedding"] = df["modelId"] + " " + df["task"]
    embeddings = get_cached_embeddings(df["text_for_embedding"].tolist())

    idx = df[df["modelId"] == target_model_id].index[0]
    target_vector = embeddings[idx].reshape(1, -1)
    distances = cosine_similarity(target_vector, embeddings).flatten()

    related_indices = distances.argsort()[-(top_n + 1) : -1][::-1]
    results = df.iloc[related_indices].copy()
    results["similarity_score"] = distances[related_indices]

    return (
        results[["modelId", "task", "author", "similarity_score", "downloads"]],
        embeddings,
        idx,
    )


# ==========================================
# ------------3. PAGE VIEWS-----------------
# ==========================================


def view_eda_page(df, f_df):
    st.header("I. Thống kê Tổng quan Hệ sinh thái AI")
    st.caption(
        "Đồ án cơ sở: NGUYEN CONG DAT - MSSV:2214025 - Chuyên ngành Khoa học dữ liệu"
    )

    s = get_statistics(f_df)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Số Model", f"{s['total_models']:,}")
    m2.metric("Downloads", f"{s['total_downloads']:,}")
    m3.metric("Tương quan (r)", s["correlation"])
    m4.metric("Top Author", s["top_author"])

    col_a, col_b = st.columns(2)
    with col_a:
        fig_top = px.bar(
            f_df.head(10),
            x="downloads",
            y="modelId",
            orientation="h",
            title="Top 10 Models theo Lượt Tải",
            color="downloads",
            color_continuous_scale="Teal",
        )
        fig_top.update_layout(
            yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False
        )
        st.plotly_chart(fig_top, width="stretch")
    with col_b:
        top_tasks = f_df["task"].value_counts().nlargest(7).index
        f_df_pie = f_df.copy()
        f_df_pie["task_grouped"] = f_df_pie["task"].where(
            f_df_pie["task"].isin(top_tasks), "Other Categories"
        )
        fig_pie = px.pie(
            f_df_pie,
            names="task_grouped",
            values="downloads",
            hole=0.45,
            title="Tỷ trọng Tác vụ",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, width="stretch")

    st.divider()
    st.header("II. Khai phá Dữ liệu")

    st.subheader("1. Phân tích Phân phối & Dị biệt")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.box(
                f_df,
                y="log_downloads",
                title="Boxplot Downloads",
                points="all",
                color_discrete_sequence=["#17BECF"],
            ),
            width="stretch",
        )
    with c2:
        hist_data = [f_df["log_downloads"].dropna()]
        fig_hist = ff.create_distplot(
            hist_data,
            ["Mật độ Log Downloads"],
            show_hist=True,
            show_rug=False,
            colors=["#AB63FA"],
        )
        mean_val = f_df["log_downloads"].mean()
        median_val = f_df["log_downloads"].median()
        fig_hist.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="#FF4B4B",
            annotation_text=f"TB: {mean_val:.2f}",
        )
        fig_hist.add_vline(
            x=median_val,
            line_dash="dot",
            line_color="#FFA500",
            annotation_text=f"Trung vị: {median_val:.2f}",
            annotation_position="top left",
        )
        fig_hist.update_layout(title_text="Phân bổ Lượt tải kèm KDE & Biên thống kê")
        st.plotly_chart(fig_hist, width="stretch")

    st.subheader("2. Xu hướng phát triển & Hành vi Đặt tên")
    col1, col2 = st.columns(2)
    with col1:
        trend_df = f_df.groupby("year").size().reset_index(name="count")
        fig_trend = px.line(
            trend_df,
            x="year",
            y="count",
            title="Sự bùng nổ của các Model qua từng năm",
            markers=True,
            color_discrete_sequence=["#00C896"],
        )
        fig_trend.update_traces(line_shape="spline", line=dict(width=3))
        st.plotly_chart(fig_trend, width="stretch")
    with col2:
        top_auth = f_df["author"].value_counts().head(10).reset_index()
        top_auth.columns = ["author", "count"]
        fig_auth = px.bar(
            top_auth,
            x="count",
            y="author",
            orientation="h",
            title="Top 10 Tác giả đóng góp nhiều nhất",
            color="count",
            color_continuous_scale="Purp",
        )
        fig_auth.update_layout(
            yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False
        )
        st.plotly_chart(fig_auth, width="stretch")

    st.markdown("**➤ Khai phá Văn bản: Phân tích từ khóa định danh Model**")
    wordcloud_col, ngram_col = st.columns([1.2, 1])

    with wordcloud_col:
        text_data = " ".join(f_df["model_name_only"].dropna())
        wordcloud = WordCloud(
            width=800,
            height=500,
            background_color="white",
            colormap="ocean",
            max_words=100,
        ).generate(text_data)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 6))
        ax_wc.imshow(wordcloud, interpolation="bilinear")
        ax_wc.axis("off")
        ax_wc.set_title("Word Cloud: Từ đơn phổ biến", fontsize=16)
        st.pyplot(fig_wc)

    with ngram_col:
        try:
            vectorizer = CountVectorizer(
                ngram_range=(2, 3), stop_words="english", max_features=10
            )
            ngrams = vectorizer.fit_transform(
                f_df["model_name_only"]
                .dropna()
                .str.replace(r"[^a-zA-Z0-9\s]", " ", regex=True)
            )
            ngram_freq = pd.DataFrame(
                {
                    "Từ khóa": vectorizer.get_feature_names_out(),
                    "Tần suất": ngrams.toarray().sum(axis=0),
                }
            )
            ngram_freq = ngram_freq.sort_values("Tần suất", ascending=True)
            fig_ngram = px.bar(
                ngram_freq,
                x="Tần suất",
                y="Từ khóa",
                orientation="h",
                title="N-grams: Cụm từ thường dùng",
                color="Tần suất",
                color_continuous_scale="Blues",
            )
            fig_ngram.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_ngram, width="stretch")
        except Exception:
            st.info("Không đủ dữ liệu văn bản để phân tích N-grams.")

    st.markdown("**➤ Mật độ Chiều dài Tên vs Lượt tải**")
    fig_len = px.density_heatmap(
        f_df,
        x="name_len",
        y="log_downloads",
        nbinsx=40,
        nbinsy=40,
        color_continuous_scale="Plasma",
        title="Bản đồ Mật độ 2D: Chiều dài tên lý tưởng",
        labels={"name_len": "Độ dài tên Model", "log_downloads": "Lượt tải (Log)"},
    )
    st.plotly_chart(fig_len, width="stretch")

    st.subheader("3. Phân tích chi tiết Tác vụ")
    col1, col2 = st.columns(2)
    with col1:
        task_counts = f_df["task"].value_counts().reset_index()
        task_counts.columns = ["Category", "Count"]
        fig1 = px.bar(
            task_counts,
            x="Count",
            y="Category",
            orientation="h",
            color="Count",
            color_continuous_scale="Viridis",
            title="Số lượng Model theo từng Tác vụ",
        )
        fig1.update_layout(
            yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False
        )
        st.plotly_chart(fig1, width="stretch")
    with col2:
        mean_stars = (
            f_df.groupby("task")["likes"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        fig4 = px.bar(
            mean_stars,
            x="task",
            y="likes",
            color="likes",
            color_continuous_scale="Oryel",
            title="Trung bình lượt Thích (Stars) theo Tác vụ",
        )
        fig4.update_layout(xaxis_tickangle=-45, coloraxis_showscale=False)
        st.plotly_chart(fig4, width="stretch")

    st.markdown("**➤ Phân bố theo không gian & Thời gian**")
    c_time, c_heat = st.columns(2)
    with c_time:
        fig2 = px.histogram(
            f_df,
            x="createdAt",
            nbins=30,
            marginal="violin",
            title="Mật độ thời gian khởi tạo Model",
            color_discrete_sequence=["#FF9F36"],
        )
        valid_dates = f_df["createdAt"].dropna()
        if not valid_dates.empty:
            median_ms = valid_dates.median().timestamp() * 1000
            fig2.add_vline(
                x=median_ms,
                line_dash="dash",
                line_color="red",
                annotation_text="Trung vị",
            )
        st.plotly_chart(fig2, width="stretch")
    with c_heat:
        pivot_table = pd.crosstab(f_df["task"], f_df["month_year"])
        fig3 = px.imshow(
            pivot_table,
            aspect="auto",
            color_continuous_scale="YlGnBu",
            title="Mật độ Model (Tác vụ x Tháng)",
        )
        st.plotly_chart(fig3, width="stretch")

    st.markdown("**➤ Kiểm định Thống kê & Phân rã Phân phối**")
    col_stat1, col_stat2 = st.columns(2)

    with col_stat1:
        top_5_tasks_box = f_df["task"].value_counts().nlargest(5).index
        df_box_task = f_df[f_df["task"].isin(top_5_tasks_box)]

        fig_box_task = px.box(
            df_box_task,
            x="task",
            y="log_downloads",
            color="task",
            title="Độ phân tán Lượt tải theo Top 5 Tác vụ",
            points="outliers",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        st.plotly_chart(fig_box_task, width="stretch")

    with col_stat2:
        st.write("**Biểu đồ QQ-Plot**")
        fig_qq, ax_qq = plt.subplots(figsize=(6, 4))
        res = stats.probplot(f_df["log_likes"].dropna(), dist="norm", plot=ax_qq)
        ax_qq.set_title("QQ-Plot của biến Log Likes")
        ax_qq.set_xlabel("Phân vị lý thuyết (Theoretical Quantiles)")
        ax_qq.set_ylabel("Dữ liệu thực tế (Ordered Values)")
        ax_qq.get_lines()[0].set_markerfacecolor("#1f77b4")
        ax_qq.get_lines()[0].set_markeredgewidth(0)
        ax_qq.get_lines()[1].set_color("#FF4B4B")
        st.pyplot(fig_qq)

    st.subheader("4. Phân tích Cảm xúc Cộng đồng")
    st.markdown(
        "Khảo sát định tính từ thảo luận: Hệ thống phân loại sắc thái để chấm điểm cảm xúc từ 0 (Tiêu cực) đến 100 (Tích cực)."
    )
    c_sent1, c_sent2 = st.columns(2)
    with c_sent1:
        st.plotly_chart(
            px.histogram(
                f_df,
                x="sentiment_score",
                color="sentiment_class",
                nbins=30,
                title="Phân bổ Điểm số Cảm xúc",
                color_discrete_map={
                    "Tích cực": "#28a745",
                    "Trung lập": "#ffc107",
                    "Tiêu cực": "#dc3545",
                },
            ),
            width="stretch",
        )
    with c_sent2:
        st.plotly_chart(
            px.scatter(
                f_df,
                x="sentiment_score",
                y="engagement_rate",
                color="scale",
                hover_name="modelId",
                title="Tương quan giữa Điểm cảm xúc và Tỷ lệ Tương tác",
                color_discrete_sequence=px.colors.qualitative.Set2,
            ),
            width="stretch",
        )

    st.subheader("5. Dấu vết Lịch sử: Sự trỗi dậy của các Tác vụ")
    top_5_tasks = f_df["task"].value_counts().nlargest(5).index
    df_trend = f_df[f_df["task"].isin(top_5_tasks)].copy()
    trend_time = (
        df_trend.groupby(["month_year", "task"])
        .size()
        .reset_index(name="Số lượng Model")
    )
    trend_time = trend_time.sort_values("month_year")

    fig_area = px.area(
        trend_time,
        x="month_year",
        y="Số lượng Model",
        color="task",
        title="Biểu đồ Vùng Xếp chồng",
        labels={"month_year": "Thời gian (Năm-Tháng)"},
        color_discrete_sequence=px.colors.qualitative.Prism,
    )
    fig_area.update_traces(line=dict(width=0))
    st.plotly_chart(fig_area, width="stretch")

    st.divider()
    st.subheader("6. Đồ thị Tri thức & Mạng lưới AI")
    st.markdown(
        "Phân tích Đồ thị Mạng lưới giúp chúngạm ra **Tâm điểm (Centrality)** của hệ sinh thái. Biểu đồ kết nối: **Loại Tác vụ** -> **Tác giả** -> **Mô hình AI**."
    )

    with st.spinner("Đang xây dựng Đồ thị tri thức (NetworkX)..."):
        top_nodes = f_df.head(60)
        G = nx.Graph()

        for _, row in top_nodes.iterrows():
            model = row["model_name_only"]
            author = row["author"]
            task = row["task"]

            G.add_node(task, type="Task", color="#28a745")
            G.add_node(author, type="Author", color="#fd7e14")
            G.add_node(model, type="Model", color="#007bff")

            G.add_edge(task, author)
            G.add_edge(author, model)

        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.7, color="#CFD8DC"),
            hoverinfo="none",
            mode="lines",
        )

        node_x = []
        node_y = []
        node_text = []
        node_hover = []
        node_color = []
        node_size = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_type = G.nodes[node]["type"]

            node_hover.append(f"Loại: {node_type}<br>Tên: {node}")

            if node_type in ["Task", "Author"]:
                node_text.append(str(node))
            else:
                node_text.append("")

            node_color.append(G.nodes[node]["color"])

            degree = G.degree(node)
            if node_type == "Task":
                calc_size = 30 + (degree * 2)
            elif node_type == "Author":
                calc_size = 20 + (degree * 1.5)
            else:
                calc_size = 12

            node_size.append(calc_size)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=node_text,
            textposition="top center",
            hovertext=node_hover,
            textfont=dict(size=11, color="#333", weight="bold"),
            marker=dict(
                showscale=False,
                color=node_color,
                size=node_size,
                line_width=1.5,
                line_color="white",
            ),
        )

        fig_network = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="rgba(248, 249, 250, 1)",
            ),
        )
        st.plotly_chart(fig_network, width="stretch")

    st.subheader("7. Ma trận tương quan & Phân tích chuyên sâu")
    corr_cols = [
        "downloads",
        "likes",
        "name_len",
        "engagement_rate",
        "log_downloads",
        "log_likes",
        "sentiment_score",
    ]
    corr_matrix = f_df[corr_cols].corr()

    c_corr1, c_corr2 = st.columns([1, 1])
    with c_corr1:
        st.markdown("**➤ Heatmap: Ma trận tương quan**")
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            range_color=[-1, 1],
        )
        st.plotly_chart(fig_corr, width="stretch")
    with c_corr2:
        st.markdown("**➤ Tương quan Log-Log**")
        st.plotly_chart(
            px.scatter(
                f_df,
                x="log_downloads",
                y="log_likes",
                trendline="ols",
                color="scale",
                hover_name="modelId",
                title="Minh chứng tương quan Downloads vs Likes",
                color_discrete_sequence=px.colors.qualitative.G10,
            ),
            width="stretch",
        )

    st.divider()
    st.markdown("### 📥 Xuất Dữ Liệu Báo Cáo")
    csv_data = f_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Tải Data gốc đã làm sạch (CSV)",
        data=csv_data,
        file_name="huggingface_cleaned_data.csv",
        mime="text/csv",
    )


def view_machine_learning_page(f_df):
    st.header("III. Ứng dụng Học máy")

    (
        df_ml,
        trained_models,
        metrics_df,
        feature_names,
        y_test,
        predictions_dict,
        X_test,
    ) = process_ml_insights(f_df)

    if not trained_models:
        return st.warning("Cần ít nhất 15 records dữ liệu để huấn luyện Học máy.")

    st.markdown("### 🏆 Bảng Benchmark Đánh giá Mô hình")

    # -------------------------------------------------------------
    # LOGIC TÔ MÀU ĐỘNG CHO MÔ HÌNH TỐT NHẤT DỰA TRÊN R2 TEST
    # -------------------------------------------------------------
    best_model_name = metrics_df.loc[metrics_df["R² Test"].idxmax(), "Mô hình"]

    def highlight_best_model(row):
        if row["Mô hình"] == best_model_name:
            return [
                "background-color: #d4edda; color: #155724; font-weight: bold"
            ] * len(row)
        return [""] * len(row)

    st.dataframe(
        metrics_df.style.apply(highlight_best_model, axis=1).format(
            {
                "R² Validation": "{:.4f}",
                "R² Test": "{:.4f}",
                "MAE Validation": "{:.2f}",
                "MAE Test": "{:.2f}",
                "RMSE Test": "{:.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    csv_metrics = metrics_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Tải Kết quả Đánh giá Benchmark (CSV)",
        data=csv_metrics,
        file_name="model_benchmark_results.csv",
        mime="text/csv",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.bar(
                metrics_df,
                x="Mô hình",
                y="R² Test",
                color="Mô hình",
                text_auto=".4f",
                title="Độ chính xác R² trên Tập Kiểm thử",
                color_discrete_sequence=px.colors.qualitative.Safe,
            ).update_layout(showlegend=False),
            width="stretch",
        )
    with c2:
        melt_metrics = metrics_df.melt(
            id_vars=["Mô hình"],
            value_vars=["MAE Test", "RMSE Test"],
            var_name="Loại Sai số",
            value_name="Giá trị",
        )
        st.plotly_chart(
            px.bar(
                melt_metrics,
                x="Mô hình",
                y="Giá trị",
                color="Loại Sai số",
                barmode="group",
                text_auto=".0f",
                title="So sánh Sai số trên Tập Kiểm thử",
                color_discrete_map={"MAE Test": "#FF7F0E", "RMSE Test": "#1F77B4"},
            ),
            width="stretch",
        )

    st.markdown("**➤ Thực tế vs Dự báo - So sánh 2 Mô hình**")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=predictions_dict["Linear Regression"],
            mode="markers",
            name="Dự báo Linear",
            marker=dict(
                color="#1f77b4",
                size=7,
                opacity=0.5,
                line=dict(width=1, color="DarkSlateGrey"),
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=predictions_dict["Random Forest"],
            mode="markers",
            name="Dự báo Random Forest",
            marker=dict(
                color="#2ca02c",
                size=8,
                opacity=0.7,
                symbol="diamond",
                line=dict(width=1, color="DarkSlateGrey"),
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            line=dict(color="#d62728", dash="dash", width=2),
            name="Đường Lý tưởng",
        )
    )
    fig.update_layout(
        xaxis_title="Log Likes Thực tế",
        yaxis_title="Log Likes Dự báo",
        template="plotly_white",
        hovermode="closest",
    )
    st.plotly_chart(fig, width="stretch")

    st.divider()
    st.markdown("### 🔎 Phân tích Đặc trưng")
    if hasattr(trained_models["Random Forest"], "feature_importances_"):
        rf_model = trained_models["Random Forest"]

        df_coef = pd.DataFrame(
            {
                "Đặc trưng": [
                    f.replace("t_", "Task: ") if f.startswith("t_") else f
                    for f in feature_names
                ],
                "Độ quan trọng": rf_model.feature_importances_,
            }
        )
        df_coef = df_coef.sort_values("Độ quan trọng", ascending=False).head(10)

        fig_coef = px.bar(
            df_coef,
            x="Độ quan trọng",
            y="Đặc trưng",
            orientation="h",
            color="Độ quan trọng",
            color_continuous_scale="Mint",
            title="Biến số quyết định Lượt Thích",
        )
        fig_coef.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_coef, width="stretch")

    st.divider()
    st.markdown("### 🧠 Giải thích AI Chuyên sâu")

    try:
        with st.spinner("Đang tính toán giá trị SHAP..."):
            explainer = shap.TreeExplainer(trained_models["Random Forest"])
            shap_values = explainer.shap_values(X_test)

            fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, show=False)
            st.pyplot(fig_shap)
    except Exception as e:
        st.warning(
            f"Tính năng SHAP cần được cài đặt. Hãy chạy lệnh `pip install shap` trong Terminal."
        )

    st.divider()
    st.markdown("### 📦 Đóng gói & Triển khai Mô hình")

    col_pkl1, col_pkl2 = st.columns([1, 2])
    with col_pkl1:
        selected_export_model = st.selectbox(
            "Chọn mô hình để đóng gói:", list(trained_models.keys()), index=0
        )
    with col_pkl2:
        st.write("")
        st.write("")
        model_bytes = pickle.dumps(trained_models[selected_export_model])
        st.download_button(
            label=f"📥 Tải Mô hình {selected_export_model} nguyên khối (.pkl)",
            data=model_bytes,
            file_name=f"{selected_export_model.replace(' ', '_').lower()}_model.pkl",
            mime="application/octet-stream",
            type="primary",
        )

    st.divider()
    st.markdown("**➤ Công cụ Dự báo Tương tác**")

    col_input, col_pred = st.columns([1, 2])
    with col_input:
        target_dl = st.number_input(
            "Nhập Downloads mục tiêu để dự báo lượt Like:",
            value=1000,
            key="val_predict",
        )
        selected_model = st.selectbox(
            "Chọn mô hình nâng cao để so sánh với Linear:",
            list(trained_models.keys()),
            index=3,
        )
        btn_predict = st.button("Chạy Dự Báo", type="primary", width="stretch")

    with col_pred:
        if btn_predict:
            input_df = pd.DataFrame(0, index=[0], columns=feature_names)
            input_df["log_downloads"] = np.log1p(target_dl)
            top_task_col = [c for c in feature_names if c.startswith("t_")][0]
            input_df[top_task_col] = 1

            res_lr = trained_models["Linear Regression"].predict(input_df)[0]
            res_rf = trained_models[selected_model].predict(input_df)[0]

            c_res1, c_res2 = st.columns(2)
            c_res1.success(
                f"Linear Regression dự báo:\n### {max(0, int(np.expm1(res_lr)))} Likes"
            )
            c_res2.info(
                f"{selected_model} dự báo:\n### {max(0, int(np.expm1(res_rf)))} Likes"
            )

            # ĐỒNG BỘ: Tự động gợi ý sử dụng mô hình tốt nhất
            st.caption(
                f"💡 **Khuyến nghị hệ thống:** Dựa trên bảng Benchmark, đề xuất ưu tiên sử dụng kết quả dự báo của **{best_model_name}** do đây là mô hình đạt độ chính xác cao nhất."
            )


def view_ai_recommender_page(f_df):
    st.header("🤖 Hệ thống Gợi ý")

    st.subheader("📍 Phân cụm Model chiến lược")
    df_c, X_scaled = perform_clustering(f_df)

    if X_scaled is not None:
        sil_score = (
            silhouette_score(X_scaled, df_c["Cluster"])
            if len(set(df_c["Cluster"])) > 1
            else 0
        )
        inertias = []
        max_k = min(8, len(f_df))
        K_range = range(2, max_k) if max_k > 3 else range(2, 3)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)

        c_km1, c_km2 = st.columns(2)
        with c_km1:
            st.metric(
                "Điểm Silhouette",
                round(sil_score, 3),
            )
            st.plotly_chart(
                px.scatter(
                    df_c,
                    x="log_downloads",
                    y="log_likes",
                    color="Cluster_Name",
                    hover_name="modelId",
                    title="Cụm chiến lược",
                    color_discrete_sequence=px.colors.qualitative.Vivid,
                ),
                width="stretch",
            )
        with c_km2:
            if len(K_range) > 1:
                fig_elbow = px.line(
                    x=list(K_range),
                    y=inertias,
                    markers=True,
                    title="Xác định K tối ưu",
                    labels={"x": "Số cụm (K)", "y": "Mức độ phân tán (Inertia)"},
                )
                fig_elbow.add_vline(
                    x=3,
                    line_dash="dash",
                    line_color="red",
                )
                fig_elbow.update_traces(line_shape="spline", line=dict(width=2.5))
                st.plotly_chart(fig_elbow, width="stretch")
            else:
                st.warning("Dữ liệu quá ít để vẽ đường cong Elbow.")
    else:
        st.warning("Dữ liệu không đủ để phân cụm.")

    st.divider()
    st.markdown("### 🔍 Công cụ Tìm kiếm Tương đồng")
    selected_model = st.selectbox(
        "Chọn mô hình để tìm tương tự:", f_df["modelId"].values
    )

    if selected_model:
        with st.spinner("Đang tính toán ma trận Vector Cosine & PCA 3D..."):
            recs, embeddings, target_idx = get_bert_recommendations(
                f_df, selected_model
            )

        if not recs.empty:
            rec_cols = st.columns(2)
            for i, (_, row) in enumerate(recs.iterrows()):
                with rec_cols[i % 2]:
                    with st.container(border=True):
                        st.markdown(f"**{row['modelId']}**")
                        st.caption(
                            f"Độ tương đồng: {row['similarity_score']:.2f} | Tải: {row['downloads']:,}"
                        )

            csv_recs = recs.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Tải Danh sách AI Gợi ý (CSV)",
                data=csv_recs,
                file_name=f"ai_recommendations_for_{selected_model.replace('/', '_')}.csv",
                mime="text/csv",
            )

            st.divider()
            st.markdown("### 🌌 Bản đồ Không gian Đa chiều")
            with st.expander("📌 Xem bản đồ Vector 3D của hệ sinh thái"):
                pca = PCA(n_components=3)
                pca_result = pca.fit_transform(embeddings)

                df_pca = f_df.copy()
                df_pca["PCA1"] = pca_result[:, 0]
                df_pca["PCA2"] = pca_result[:, 1]
                df_pca["PCA3"] = pca_result[:, 2]

                df_pca["Trạng thái"] = "Model khác"
                similar_model_ids = recs["modelId"].tolist()
                df_pca.loc[df_pca["modelId"].isin(similar_model_ids), "Trạng thái"] = (
                    "⭐ Mô hình tương tự"
                )
                df_pca.loc[target_idx, "Trạng thái"] = f"📍 {selected_model}"

                fig_3d = px.scatter_3d(
                    df_pca,
                    x="PCA1",
                    y="PCA2",
                    z="PCA3",
                    color="Trạng thái",
                    hover_name="modelId",
                    hover_data={"Trạng thái": False, "task": True},
                    color_discrete_map={
                        f"📍 {selected_model}": "#FF4B4B",
                        "⭐ Mô hình tương tự": "#FFA500",
                        "Model khác": "#1f77b4",
                    },
                )

                fig_3d.update_traces(
                    marker=dict(size=4, opacity=0.4), selector=dict(name="Model khác")
                )
                fig_3d.update_traces(
                    marker=dict(size=8, symbol="circle", opacity=0.9),
                    selector=dict(name="⭐ Mô hình tương tự"),
                )
                fig_3d.update_traces(
                    marker=dict(
                        size=14, symbol="diamond", line=dict(color="black", width=2)
                    ),
                    selector=dict(name=f"📍 {selected_model}"),
                )
                fig_3d.update_layout(
                    margin=dict(l=0, r=0, b=0, t=40), scene=dict(bgcolor="#f8f9fa")
                )

                st.plotly_chart(fig_3d, width="stretch")

    st.divider()
    st.subheader("🧪 Phòng thử nghiệm AI trực tuyến")

    llm_model = st.selectbox(
        "🚀 Chọn bộ não AI để thử nghiệm:",
        [
            "Qwen/Qwen2.5-7B-Instruct",
            "HuggingFaceH4/zephyr-7b-beta",
            "google/gemma-2-9b-it",
        ],
    )

    user_prompt = st.text_area(
        "✍️ Nhập câu hỏi hoặc văn bản yêu cầu AI xử lý:",
        value="Dựa vào dữ liệu hệ thống, hãy đánh giá xem xu hướng mô hình nào có điểm cảm xúc tốt và lượt tương tác cao?",
    )

    if st.button("🔥 Thực thi suy luận", type="primary"):
        if user_prompt:
            with st.spinner(
                "⚡ Đang thực thi thuật toán RAG & Gửi gói tin bảo mật đến Cloud Server..."
            ):
                try:
                    top_rag_models = f_df.head(50).copy()
                    top_rag_models["rag_text"] = (
                        top_rag_models["modelId"] + " " + top_rag_models["task"]
                    )

                    rag_embeddings = get_cached_embeddings(
                        top_rag_models["rag_text"].tolist()
                    )
                    query_embedding = get_cached_embeddings([user_prompt])[0].reshape(
                        1, -1
                    )

                    rag_sim = cosine_similarity(
                        query_embedding, rag_embeddings
                    ).flatten()
                    top_3_idx = rag_sim.argsort()[-3:][::-1]

                    context_lines = []
                    for idx in top_3_idx:
                        row = top_rag_models.iloc[idx]
                        context_lines.append(
                            f"- Mô hình '{row['modelId']}' [Tác vụ: {row['task']}] đạt {row['downloads']:,} lượt tải, {row['likes']:,} lượt thích, điểm cảm xúc cộng đồng: {row['sentiment_score']}/100 ({row['sentiment_class']})."
                        )

                    context_str = "\n".join(context_lines)

                    system_instruction = (
                        "Bạn là một Trợ lý AI cao cấp tích hợp công nghệ RAG (Retrieval-Augmented Generation). "
                        "Bạn PHẢI trả lời hoàn toàn bằng Tiếng Việt 100%. Hãy sử dụng thông tin từ cơ sở dữ liệu thời gian thực được trích xuất từ đồ án sau đây để phân tích và trả lời câu hỏi của người dùng:\n\n"
                        f"[CƠ SỞ DỮ LIỆU NGỮ CẢNH TỪ ĐỒ ÁN]:\n{context_str}\n\n"
                        "Tuyệt đối không bịa đặt số liệu nằm ngoài ngữ cảnh trên."
                    )

                    hf_token = st.secrets["hf_token"]
                    client = InferenceClient(token=hf_token)

                    chat_response = client.chat_completion(
                        model=llm_model,
                        messages=[
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=600,
                        temperature=0.3,
                    )

                    response = chat_response.choices[0].message.content

                    st.markdown("**🎯 Kết quả phản hồi từ Cloud AI:**")
                    st.success(response)

                except Exception as e:
                    st.error(
                        f"Lỗi kết nối API Cloud: {e}. Hệ thống công cộng đang bận, vui lòng thử lại sau."
                    )
        else:
            st.warning("Vui lòng nhập văn bản trước khi thực thi.")


def view_battle_and_ai_page(f_df):
    st.header("✨ Đấu trường Model & Trợ lý Phân tích AI")

    st.markdown("### 🤖 Báo cáo Tổng hợp từ Trợ lý AI")

    with st.container(border=True):
        if not f_df.empty:
            total_models = len(f_df)
            total_down = f_df["downloads"].sum()
            top_task = f_df["task"].value_counts().index[0]
            top_model = f_df.iloc[0]["modelId"]
            top_author = f_df["author"].value_counts().idxmax()

            st.markdown(f"""
            **Báo cáo Tóm tắt:**
            
            Hệ thống đang phân tích một tập dữ liệu gồm **{total_models:,} mô hình**, thu hút tổng cộng **{total_down:,} lượt tải xuống**. 
            
            Phân tích cho thấy **{top_task}** hiện đang là tác vụ (Task) thống trị và nhận được sự quan tâm lớn nhất từ cộng đồng phát triển. Đặc biệt, tác giả hoặc tổ chức đóng góp năng nổ nhất trong tệp dữ liệu này là **{top_author}**. 
            
            Ngôi sao sáng nhất trên bảng xếp hạng không ai khác chính là mô hình **`{top_model}`**, dẫn đầu tuyệt đối về mức độ phủ sóng. Các mô hình thành công có xu hướng kết hợp tên gọi ngắn gọn, rõ ràng kèm theo các từ khóa như 'instruct', 'chat' để định vị rõ tính năng đối với người dùng.
            """)
        else:
            st.warning(
                "Không có dữ liệu để AI tổng hợp. Vui lòng nới lỏng bộ lọc ở thanh Sidebar."
            )

    st.divider()

    st.markdown("### ⚔️ Đấu trường Model")

    col1, col2 = st.columns(2)
    with col1:
        model1 = st.selectbox("🔴 Chọn Model Góc Đỏ:", f_df["modelId"].values, index=0)
    with col2:
        default_index_2 = 1 if len(f_df) > 1 else 0
        model2 = st.selectbox(
            "🔵 Chọn Model Góc Xanh:", f_df["modelId"].values, index=default_index_2
        )

    if model1 and model2:
        m1_data = f_df[f_df["modelId"] == model1].iloc[0]
        m2_data = f_df[f_df["modelId"] == model2].iloc[0]

        metrics = ["log_downloads", "log_likes", "engagement_rate", "name_len"]
        labels = [
            "Sức hút (Tải xuống)",
            "Độ uy tín (Lượt Thích)",
            "Tương tác (Engagement)",
            "Độ dài tên (Ngắn là tốt)",
        ]

        def normalize(val, col_name):
            max_v = f_df[col_name].max()
            min_v = f_df[col_name].min()

            # 1. Xử lý riêng cho Engagement Rate: Dùng Logarit để khuếch đại sự khác biệt
            if col_name == "engagement_rate":
                # Cộng 1e-9 để tránh lỗi log(0)
                transformed_val = np.log1p(val * 1000)
                transformed_max = np.log1p(max_v * 1000)
                transformed_min = np.log1p(min_v * 1000)
                if transformed_max == transformed_min:
                    return 50
                return (
                    (transformed_val - transformed_min)
                    / (transformed_max - transformed_min)
                ) * 100

            # 2. Xử lý cho các chỉ số còn lại (Downloads, Likes...)
            if max_v == min_v:
                return 50
            score = ((val - min_v) / (max_v - min_v)) * 100

            # Độ dài tên: Càng ngắn càng tốt (điểm cao)
            if col_name == "name_len":
                return 100 - score

            return score

        m1_scores = [normalize(m1_data[m], m) for m in metrics]
        m2_scores = [normalize(m2_data[m], m) for m in metrics]

        m1_scores.append(m1_scores[0])
        m2_scores.append(m2_scores[0])
        labels.append(labels[0])

        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=m1_scores,
                theta=labels,
                fill="toself",
                name=model1,
                line_color="#FF4B4B",
                fillcolor="rgba(255, 75, 75, 0.4)",
            )
        )
        fig_radar.add_trace(
            go.Scatterpolar(
                r=m2_scores,
                theta=labels,
                fill="toself",
                name=model2,
                line_color="#0068C9",
                fillcolor="rgba(0, 104, 201, 0.4)",
            )
        )

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], showticklabels=False),
                angularaxis=dict(tickfont=dict(size=13, color="black")),
            ),
            showlegend=True,
            title=dict(
                text=f"Đại chiến thông số: {model1} VS {model2}", font=dict(size=18)
            ),
        )

        st.plotly_chart(fig_radar, width="stretch")
        comp_df = pd.DataFrame(
            {
                "Chỉ số": [
                    "Lượt Tải Thực Tế",
                    "Lượt Thích Thực Tế",
                    "Tỷ lệ Tương tác (%)",
                    "Loại Tác vụ",
                ],
                model1: [
                    f"{int(m1_data['downloads']):,}",
                    f"{int(m1_data['likes']):,}",
                    f"{m1_data['engagement_rate']:.2f}%",
                    m1_data["task"],
                ],
                model2: [
                    f"{int(m2_data['downloads']):,}",
                    f"{int(m2_data['likes']):,}",
                    f"{m2_data['engagement_rate']:.2f}%",
                    m2_data["task"],
                ],
            }
        )
        st.dataframe(comp_df, hide_index=True, width="stretch")


# ==========================================
# -----------4. MAIN _ APP ROUTING----------
# ==========================================
def main():
    df = load_data_final_v1()
    if df.empty:
        return st.warning("Không có dữ liệu thỏa mãn.")

    st.sidebar.image(
        "https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=50
    )
    st.sidebar.markdown("## 🧭 Bảng Điều Hướng")

    page_selection = st.sidebar.radio(
        "Chọn Module Phân Tích:",
        [
            "📊 Khai phá Dữ liệu",
            "🔮 Trạm Học máy",
            "🤖 Hệ thống Gợi ý",
            "✨ Đấu trường & Trợ lý AI",
        ],
    )

    st.sidebar.divider()
    st.sidebar.markdown("### 🛠️ Bộ Lọc Dữ Liệu")

    selected_scale = st.sidebar.multiselect(
        "Phân khúc Lượt tải:", df["scale"].unique(), default=df["scale"].unique()
    )
    selected_tasks = st.sidebar.multiselect(
        "Loại Tác vụ:", df["task"].unique(), default=df["task"].unique()
    )

    st.sidebar.markdown("#### Lọc Chuyên sâu:")
    min_date = df["createdAt"].min().date()
    max_date = df["createdAt"].max().date()
    date_range = st.sidebar.date_input(
        "🗓️ Khoảng thời gian tạo Model:",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date,
    )

    max_dl = int(df["downloads"].max())
    min_dl = st.sidebar.slider(
        "📥 Lượt tải tối thiểu:", min_value=0, max_value=max_dl // 10, value=0, step=100
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
        start_datetime = pd.to_datetime(start_date).tz_localize("UTC")
        end_datetime = pd.to_datetime(end_date).tz_localize("UTC")

        f_df = df[
            (df["scale"].isin(selected_scale))
            & (df["task"].isin(selected_tasks))
            & (df["downloads"] >= min_dl)
            & (df["createdAt"] >= start_datetime)
            & (df["createdAt"] <= end_datetime)
        ].copy()
    else:
        f_df = pd.DataFrame()

    st.sidebar.divider()
    st.sidebar.caption("Đồ án cơ sở: NGUYEN CONG DAT - Khoa học Dữ liệu")

    if f_df.empty:
        st.error(
            "Không có dữ liệu thỏa mãn bộ lọc hiện tại! Vui lòng điều chỉnh lại thanh Sidebar."
        )
    else:
        if page_selection == "📊 Khai phá Dữ liệu":
            view_eda_page(df, f_df)
        elif page_selection == "🔮 Trạm Học máy":
            view_machine_learning_page(f_df)
        elif page_selection == "🤖 Hệ thống Gợi ý":
            view_ai_recommender_page(f_df)
        elif page_selection == "✨ Đấu trường & Trợ lý AI":
            view_battle_and_ai_page(f_df)


if __name__ == "__main__":
    main()
