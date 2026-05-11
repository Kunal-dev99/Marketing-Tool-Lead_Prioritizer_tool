import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Lead Intelligence Command Center",
    layout="wide"
)

# =========================================================
# HEADER
# =========================================================

st.title("Lead Intelligence Command Center")
st.caption("AI-Assisted Marketing Intelligence & Lead Prioritization")

# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Leads File (.xlsx or .csv)",
    type=["xlsx", "csv"]
)

# =========================================================
# MAIN PROCESSING
# =========================================================

if uploaded_file:

    # =====================================================
    # LOAD DATA
    # =====================================================

    try:

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        else:
            df = pd.read_excel(uploaded_file)

    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    st.success("File Loaded Successfully")

    # =====================================================
    # COLUMN STANDARDIZATION
    # =====================================================

    df.columns = [
        col.strip().lower().replace(" ", "_")
        for col in df.columns
    ]

    # =====================================================
    # EMAIL HELPERS
    # =====================================================

    personal_domains = [
        "gmail.com",
        "yahoo.com",
        "hotmail.com",
        "outlook.com"
    ]

    def extract_domain(email):

        try:
            return str(email).split("@")[1].lower()

        except:
            return "unknown"

    def is_corporate_email(email):

        domain = extract_domain(email)

        return domain not in personal_domains

    # =====================================================
    # PRIORITY ENGINE
    # =====================================================

    def calculate_lead_score(row):

        score = 0

        # ================================================
        # QUALIFICATION SIGNALS
        # ================================================

        email = str(row.get("email", ""))

        if is_corporate_email(email):
            score += 25

        if pd.notna(row.get("company", None)):
            score += 15

        if pd.notna(row.get("job_title", None)):
            score += 10

        if pd.notna(row.get("linkedin", None)):
            score += 10

        # ================================================
        # ENGAGEMENT SIGNALS
        # ================================================

        try:

            time_spent = float(
                row.get("average_time_spent", 0)
            )

            if time_spent >= 5:
                score += 25

            elif time_spent >= 3:
                score += 15

            elif time_spent >= 1:
                score += 5

        except:
            pass

        # ================================================
        # LEAD SOURCE SIGNALS
        # ================================================

        lead_source = str(
            row.get("lead_source", "")
        ).lower()

        if "website" in lead_source:
            score += 15

        if "referral" in lead_source:
            score += 20

        if "linkedin" in lead_source:
            score += 10

        # ================================================
        # ROLE SIGNALS
        # ================================================

        role = str(
            row.get("job_title", "")
        ).lower()

        important_roles = [
            "manager",
            "director",
            "head",
            "ceo",
            "cto",
            "founder",
            "lead"
        ]

        if any(role_name in role for role_name in important_roles):
            score += 20

        # ================================================
        # NEGATIVE SIGNALS
        # ================================================

        if not is_corporate_email(email):
            score -= 15

        if pd.isna(row.get("company", None)):
            score -= 20

        return max(score, 0)

    # =====================================================
    # PRIORITY LABEL
    # =====================================================

    def assign_priority(score):

        if score >= 80:
            return "🔥 Immediate Attention"

        elif score >= 60:
            return "🟢 High Intent"

        elif score >= 40:
            return "🟡 Warm"

        else:
            return "🔴 Low Priority"

    # =====================================================
    # PERSONA ENGINE
    # =====================================================

    def get_persona(score, corporate):

        if score >= 80 and corporate:
            return "Enterprise Decision Maker"

        elif score >= 60:
            return "Fast Moving Prospect"

        elif score >= 40:
            return "Warm Opportunity"

        else:
            return "Passive / Cold Lead"

    # =====================================================
    # ACTION ENGINE
    # =====================================================

    def next_best_action(priority):

        if "Immediate" in priority:
            return "Call within 24 hours"

        elif "High Intent" in priority:
            return "Send personalized sales email"

        elif "Warm" in priority:
            return "Add to nurturing sequence"

        else:
            return "Long-term engagement campaign"

    # =====================================================
    # EXPLAINABILITY ENGINE
    # =====================================================

    def generate_reason(row):

        reasons = []

        if is_corporate_email(str(row.get("email", ""))):
            reasons.append("Corporate email")

        if pd.notna(row.get("company", None)):
            reasons.append("Company available")

        if pd.notna(row.get("linkedin", None)):
            reasons.append("LinkedIn available")

        try:

            time_spent = float(
                row.get("average_time_spent", 0)
            )

            if time_spent >= 5:
                reasons.append("High engagement")

            elif time_spent >= 3:
                reasons.append("Moderate engagement")

        except:
            pass

        role = str(
            row.get("job_title", "")
        ).lower()

        if any(
            title in role
            for title in [
                "manager",
                "director",
                "ceo",
                "founder"
            ]
        ):
            reasons.append("Decision-maker role")

        return ", ".join(reasons)

    # =====================================================
    # LEAD PROCESSING
    # =====================================================

    df["lead_score"] = df.apply(
        calculate_lead_score,
        axis=1
    )

    df["priority"] = df["lead_score"].apply(
        assign_priority
    )

    df["corporate_email"] = df["email"].astype(
        str
    ).apply(is_corporate_email)

    df["persona"] = df.apply(
        lambda row: get_persona(
            row["lead_score"],
            row["corporate_email"]
        ),
        axis=1
    )

    df["suggested_action"] = df["priority"].apply(
        next_best_action
    )

    df["why_important"] = df.apply(
        generate_reason,
        axis=1
    )

    # =====================================================
    # METRICS
    # =====================================================

    total_leads = len(df)

    high_intent = len(
        df[df["lead_score"] >= 60]
    )

    immediate_attention = len(
        df[df["lead_score"] >= 80]
    )

    warm_leads = len(
        df[
            (df["lead_score"] >= 40) &
            (df["lead_score"] < 60)
        ]
    )

    low_priority = len(
        df[df["lead_score"] < 40]
    )

    # =====================================================
    # DASHBOARD METRICS
    # =====================================================

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Leads", total_leads)
    col2.metric("🔥 Immediate", immediate_attention)
    col3.metric("🟢 High Intent", high_intent)
    col4.metric("🟡 Warm", warm_leads)
    col5.metric("🔴 Low", low_priority)

    st.divider()

    # =====================================================
    # AI ALERTS
    # =====================================================

    st.subheader("AI Alerts")

    alert_col1, alert_col2 = st.columns(2)

    corporate_count = df["corporate_email"].sum()

    missing_company = df["company"].isna().sum()

    avg_score = round(
        df["lead_score"].mean(),
        2
    )

    with alert_col1:

        st.success(
            f"{corporate_count} leads use corporate domains"
        )

        st.info(
            f"Average lead quality score: {avg_score}"
        )

    with alert_col2:

        st.warning(
            f"{missing_company} leads missing company info"
        )

        st.error(
            "Low-quality leads should move to nurture campaigns"
        )

    st.divider()

    # =====================================================
    # PRIORITY LEAD CARDS
    # =====================================================

    st.subheader("Today's Priority Queue")

    priority_df = df.sort_values(
        by="lead_score",
        ascending=False
    )

    top_leads = priority_df.head(10)

    for idx, row in top_leads.iterrows():

        with st.container(border=True):

            col1, col2 = st.columns([3, 1])

            with col1:

                lead_name = row.get(
                    "name",
                    "Unknown Lead"
                )

                company = row.get(
                    "company",
                    "Unknown Company"
                )

                st.markdown(
                    f"### {lead_name}"
                )

                st.caption(
                    f"{row['persona']} • {company}"
                )

                st.write(
                    f"### {row['priority']}"
                )

                st.write(
                    f"**Suggested Action:** "
                    f"{row['suggested_action']}"
                )

            with col2:

                st.metric(
                    "Lead Score",
                    row["lead_score"]
                )

            st.markdown(
                "#### Why This Lead Matters"
            )

            st.info(row["why_important"])

    st.divider()

    # =====================================================
    # LEAD BEHAVIOR UNIVERSE
    # =====================================================

    st.subheader("Lead Behavior Universe")
    st.caption(
        "AI-generated behavioral similarity mapping"
    )

    clustering_features = []

    possible_features = [
        "average_time_spent",
        "lead_score"
    ]

    for feature in possible_features:

        if feature in df.columns:
            clustering_features.append(feature)

    if len(clustering_features) >= 1:

        cluster_df = df[
            clustering_features
        ].fillna(0)

        scaler = StandardScaler()

        scaled_data = scaler.fit_transform(
            cluster_df
        )

        n_samples = len(cluster_df)
        n_clusters = min(4, n_samples) if n_samples > 0 else 1

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42
        )

        clusters = kmeans.fit_predict(
            scaled_data
        )

        df["cluster"] = clusters

        # Safely determine 2D projection coordinates
        if scaled_data.shape[1] >= 2 and n_samples > 1:

            try:

                perplexity_val = min(10, n_samples - 1)
                if perplexity_val < 1:
                    perplexity_val = 1

                init_method = 'pca' if n_samples >= 2 else 'random'

                tsne = TSNE(
                    n_components=2,
                    perplexity=perplexity_val,
                    random_state=42,
                    init=init_method
                )

                tsne_results = tsne.fit_transform(
                    scaled_data
                )

                x_coords = tsne_results[:, 0]
                y_coords = tsne_results[:, 1]

            except Exception as e:

                x_coords = scaled_data[:, 0]
                y_coords = scaled_data[:, 1] if scaled_data.shape[1] > 1 else np.zeros(n_samples)

        else:

            # 1D feature fallback or single sample fallback: map along X-axis
            x_coords = scaled_data[:, 0] if n_samples > 0 else np.zeros(1)
            y_coords = np.zeros(n_samples) if n_samples > 0 else np.zeros(1)

        tsne_df = pd.DataFrame({
            "x": x_coords,
            "y": y_coords,
            "cluster": clusters
        })

        st.scatter_chart(
            tsne_df,
            x="x",
            y="y",
            color="cluster"
        )

        # =================================================
        # CLUSTER INSIGHTS
        # =================================================

        st.subheader("Cluster Intelligence")

        cluster_names = {
            0: "Enterprise Decision Makers",
            1: "Fast Moving Prospects",
            2: "Warm Opportunities",
            3: "Cold Leads"
        }

        for cluster_id in sorted(
            df["cluster"].unique()
        ):

            cluster_data = df[
                df["cluster"] == cluster_id
            ]

            cluster_name = cluster_names.get(
                cluster_id,
                f"Cluster {cluster_id}"
            )

            avg_cluster_score = round(
                cluster_data["lead_score"].mean(),
                2
            )

            with st.expander(cluster_name):

                st.write(
                    f"Total Leads: {len(cluster_data)}"
                )

                st.write(
                    f"Average Score: "
                    f"{avg_cluster_score}"
                )

                if avg_cluster_score >= 70:

                    st.success(
                        "Recommended Action: "
                        "Immediate sales outreach"
                    )

                elif avg_cluster_score >= 40:

                    st.info(
                        "Recommended Action: "
                        "Personalized nurturing"
                    )

                else:

                    st.warning(
                        "Recommended Action: "
                        "Long-term engagement"
                    )

                display_cols = [
                    col for col in [
                        "name",
                        "company",
                        "lead_score",
                        "priority",
                        "persona",
                        "suggested_action"
                    ]
                    if col in cluster_data.columns
                ]

                st.dataframe(
                    cluster_data[
                        display_cols
                    ].head(10),
                    use_container_width=True
                )

    st.divider()

    # =====================================================
    # DATA QUALITY ANALYSIS
    # =====================================================

    st.subheader("Data Quality Insights")

    quality_data = []

    for col in df.columns:

        missing = df[col].isna().sum()

        quality_data.append({
            "Column": col,
            "Missing Values": missing,
            "Missing %": round(
                (missing / len(df)) * 100,
                2
            )
        })

    quality_df = pd.DataFrame(
        quality_data
    )

    st.dataframe(
        quality_df,
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # SEARCH ASSISTANT
    # =====================================================

    st.subheader("Lead Search Assistant")

    query = st.text_input(
        "Ask something about leads"
    )

    if query:

        query = query.lower()

        if "high" in query:

            result = df[
                df["lead_score"] >= 60
            ]

            st.success(
                "Showing High Intent Leads"
            )

            st.dataframe(
                result.head(20),
                use_container_width=True
            )

        elif "warm" in query:

            result = df[
                (df["lead_score"] >= 40) &
                (df["lead_score"] < 60)
            ]

            st.success(
                "Showing Warm Leads"
            )

            st.dataframe(
                result.head(20),
                use_container_width=True
            )

        elif "company" in query:

            if "company" in df.columns:

                grouped = df.groupby(
                    "company"
                ).size().reset_index(
                    name="lead_count"
                )

                st.dataframe(
                    grouped,
                    use_container_width=True
                )

        else:

            st.info(
                "No matching insight found"
            )

    st.divider()

    # =====================================================
    # XLSX EXPORT
    # =====================================================

    st.subheader("Export Lead Intelligence Report")

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        # ================================================
        # PRIORITY LEADS
        # ================================================

        immediate_df = df[
            df["lead_score"] >= 80
        ]

        immediate_df.to_excel(
            writer,
            sheet_name="Immediate Attention",
            index=False
        )

        # ================================================
        # HIGH INTENT
        # ================================================

        high_df = df[
            (df["lead_score"] >= 60) &
            (df["lead_score"] < 80)
        ]

        high_df.to_excel(
            writer,
            sheet_name="High Intent",
            index=False
        )

        # ================================================
        # WARM LEADS
        # ================================================

        warm_df = df[
            (df["lead_score"] >= 40) &
            (df["lead_score"] < 60)
        ]

        warm_df.to_excel(
            writer,
            sheet_name="Warm Leads",
            index=False
        )

        # ================================================
        # DATA QUALITY
        # ================================================

        quality_df.to_excel(
            writer,
            sheet_name="Data Quality",
            index=False
        )

        # ================================================
        # CLUSTER INTELLIGENCE
        # ================================================

        cluster_export = df[
            [
                "cluster",
                "persona",
                "lead_score",
                "priority",
                "suggested_action"
            ]
        ]

        cluster_export.to_excel(
            writer,
            sheet_name="Behavior Clusters",
            index=False
        )

        # ================================================
        # EXECUTIVE SUMMARY
        # ================================================

        summary_df = pd.DataFrame({

            "Metric": [
                "Total Leads",
                "Immediate Attention",
                "High Intent",
                "Warm Leads",
                "Low Priority",
                "Average Lead Score"
            ],

            "Value": [
                total_leads,
                immediate_attention,
                high_intent,
                warm_leads,
                low_priority,
                avg_score
            ]
        })

        summary_df.to_excel(
            writer,
            sheet_name="Executive Summary",
            index=False
        )

    excel_data = output.getvalue()

    st.download_button(
        label="Download Lead Intelligence Report",
        data=excel_data,
        file_name="lead_intelligence_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:

    st.info(
        "Upload a lead file to begin analysis"
    )