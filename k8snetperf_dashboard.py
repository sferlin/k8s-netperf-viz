# k8snetperf_viz.py
#
# Streamlit dashboard for k8s-netperf benchmark analysis.
# Consumes the single k8snetperf_raw.csv produced by k8s-netperf.
# ToDo: Consumes the raw data queried from opensearch with orion
#
# Run:
#   pip install streamlit pandas plotly
#   streamlit run k8snetperf_viz.py
#
# ------------------------------------------------------------

import argparse
import re
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import requests
import streamlit as st

parser = argparse.ArgumentParser(
    description="Streamlit dashboard for k8s-netperf benchmark analysis.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""\
data input (via the sidebar once the app is running):
  Build log URL    Paste a CI build-log.txt URL to fetch and parse
  File upload      Upload a k8snetperf_raw.csv or build-log.txt file
  Local CSV        Place k8snetperf_raw.csv in the working directory

examples:
  streamlit run k8snetperf_dashboard.py
  streamlit run k8snetperf_dashboard.py -- --server.address 0.0.0.0 --server.port 8080
  streamlit run k8snetperf_dashboard.py -- -h
""",
)
parser.parse_known_args(sys.argv[1:])


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Netperf Benchmark Dashboard",
    layout="wide",
)

st.title("Netperf Benchmark Dashboard")

DEFAULT_CSV = "k8snetperf_raw.csv"


# ============================================================
# CSV PARSING
# ============================================================

def split_raw_csv(text):
    """Split the CSV into sections based on header rows.
    """
    lines = text.splitlines()
    sections = []
    current = []

    header_sentinels = {"Role", "Type", "Driver"}

    for line in lines:
        first_field = line.split(",", 1)[0].strip()
        if first_field in header_sentinels and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("\n".join(current))

    return sections


def normalize_bool(v):
    if isinstance(v, bool):
        return v
    if pd.isna(v):
        return False
    return str(v).strip().lower() == "true"


BOOL_COLS = ["Host Network", "VM mode", "Service", "External Server", "Same node"]


def parse_node_cpu(csv_text):
    df = pd.read_csv(StringIO(csv_text))
    for c in BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].apply(normalize_bool)
    df["Message Size"] = pd.to_numeric(df["Message Size"], errors="coerce")
    df["Parallelism"] = pd.to_numeric(df["Parallelism"], errors="coerce")
    return df


def parse_reliability(csv_text):
    df = pd.read_csv(StringIO(csv_text))
    df.rename(columns={"Type": "Reliability Type"}, inplace=True)
    for c in BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].apply(normalize_bool)
    df["Message Size"] = pd.to_numeric(df["Message Size"], errors="coerce")
    df["Parallelism"] = pd.to_numeric(df["Parallelism"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    return df


def parse_summary(csv_text):
    df = pd.read_csv(StringIO(csv_text))
    for c in BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].apply(normalize_bool)
    df["Message Size"] = pd.to_numeric(df["Message Size"], errors="coerce")
    df["Parallelism"] = pd.to_numeric(df["Parallelism"], errors="coerce")
    df["Avg Throughput"] = pd.to_numeric(df["Avg Throughput"], errors="coerce")
    if "99%tile Observed Latency" in df.columns:
        df["99%tile Observed Latency"] = pd.to_numeric(
            df["99%tile Observed Latency"], errors="coerce"
        )
    df["Confidence metric - low"] = pd.to_numeric(
        df["Confidence metric - low"], errors="coerce"
    )
    df["Confidence metric - high"] = pd.to_numeric(
        df["Confidence metric - high"], errors="coerce"
    )
    return df


def classify_section(csv_text):
    """Return a tag describing which table this section is."""
    header = csv_text.split("\n", 1)[0]
    first_field = header.split(",", 1)[0].strip()

    if first_field == "Type":
        return "reliability"
    if first_field == "Driver":
        return "summary"
    if first_field == "Role":
        if "Pod Name" in header:
            if "Utilization" in header:
                second_row = csv_text.split("\n", 2)[1] if "\n" in csv_text else ""
                vals = second_row.split(",")
                if vals:
                    try:
                        util_val = float(vals[-1])
                        if util_val > 10000:
                            return "pod_memory"
                    except (ValueError, IndexError):
                        pass
                return "pod_cpu"
        return "node_cpu"
    return "unknown"


def load_raw_csv(path):
    text = Path(path).read_text()
    return parse_text(text)


def parse_text(text):
    if re.search(r"^\+---", text, re.MULTILINE):
        return load_build_log(text)
    return _parse_csv_text(text)


def _parse_csv_text(text):
    sections = split_raw_csv(text)

    node_cpu_df = pd.DataFrame()
    reliability_df = pd.DataFrame()
    summary_df = pd.DataFrame()

    for section in sections:
        tag = classify_section(section)
        if tag == "node_cpu":
            node_cpu_df = parse_node_cpu(section)
        elif tag == "reliability":
            reliability_df = parse_reliability(section)
        elif tag == "summary":
            summary_df = parse_summary(section)

    return summary_df, reliability_df, node_cpu_df, pd.DataFrame(), pd.DataFrame()


# ============================================================
# BUILD-LOG (ASCII TABLE) PARSING
# ============================================================

def extract_ascii_tables(text):
    """Extract ASCII table blocks from build-log text.

    Tables use +---+ separator lines and | ... | data rows. Adjacent
    tables (back-to-back +---+ lines with no data row between them)
    are split into separate blocks.
    """
    table_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("+---") or (stripped.startswith("|") and stripped.endswith("|")):
            table_lines.append(stripped)
        else:
            if table_lines:
                table_lines.append(None)
    if table_lines and table_lines[-1] is not None:
        table_lines.append(None)

    tables = []
    current = []
    prev_was_separator = False
    for line in table_lines:
        if line is None:
            if current:
                data_rows = [l for l in current if l.startswith("|")]
                if len(data_rows) >= 2:
                    tables.append(current)
            current = []
            prev_was_separator = False
            continue

        is_separator = line.startswith("+---")
        if is_separator and prev_was_separator and current:
            data_rows = [l for l in current if l.startswith("|")]
            if len(data_rows) >= 2:
                tables.append(current)
            current = [line]
        else:
            current.append(line)
        prev_was_separator = is_separator

    return tables


def parse_ascii_table(lines):
    """Parse a single ASCII table block into a DataFrame."""
    data_rows = [l for l in lines if l.startswith("|")]
    if not data_rows:
        return pd.DataFrame()

    def split_row(row):
        return [cell.strip() for cell in row.strip("|").split("|")]

    headers = split_row(data_rows[0])
    rows = [split_row(r) for r in data_rows[1:]]
    return pd.DataFrame(rows, columns=headers)


COLUMN_MAP = {
    "SCENARIO": "Profile",
    "VIRT MODE": "VM mode",
    "HOST NETWORK": "Host Network",
    "SAME NODE": "Same node",
    "EXTERNAL SERVER": "External Server",
    "MESSAGE SIZE": "Message Size",
    "PARALLELISM": "Parallelism",
    "SAMPLES": "# of Samples",
    "DRIVER": "Driver",
    "DURATION": "Duration",
    "BURST": "Burst",
    "SERVICE": "Service",
    "UDN INFO": "UDN Info",
    "BRIDGE INFO": "Bridge Info",
}


def parse_value_with_units(s):
    """Extract the numeric value from strings like '1154.436000 (Mb/s)'."""
    s = s.strip()
    m = re.match(r"(-?[\d.]+)", s)
    return float(m.group(1)) if m else np.nan


def parse_confidence_interval(s):
    """Split '1095.07-1213.79 (Mb/s)' into (low, high) floats.

    Handles negative lows like '-350.75-1506.32 (Mb/s)' by splitting
    on '-' that is preceded by a digit.
    """
    s = re.sub(r"\(.*\)", "", s).strip()
    if not s:
        return np.nan, np.nan
    parts = re.split(r"(?<=\d)-", s, maxsplit=1)
    if len(parts) == 2:
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            return np.nan, np.nan
    return np.nan, np.nan


def _build_summary_from_table(df):
    """Convert a Stream/RR ASCII table DataFrame to summary schema."""
    result = pd.DataFrame()
    for old, new in COLUMN_MAP.items():
        if old in df.columns:
            result[new] = df[old]

    if "AVG VALUE" in df.columns:
        result["Avg Throughput"] = df["AVG VALUE"].apply(parse_value_with_units)

    # Same stream throughput data (TCP_STREAM / UDP_STREAM) with error bars Confidence metric - low to Confidence metric - high)
    if "95% CONFIDENCE INTERVAL" in df.columns:
        ci = df["95% CONFIDENCE INTERVAL"].apply(parse_confidence_interval)
        result["Confidence metric - low"] = ci.apply(lambda x: x[0])
        result["Confidence metric - high"] = ci.apply(lambda x: x[1])

    return result


def load_build_log(text):
    """Parse ASCII tables from a build log into the same DataFrames as CSV."""
    raw_tables = extract_ascii_tables(text)
    parsed = [parse_ascii_table(t) for t in raw_tables]

    stream_parts = []
    rr_parts = []
    latency_parts = []
    reliability_parts = []

    for df in parsed:
        if df.empty:
            continue
        first_col = df.columns[0]

        if first_col == "RESULT TYPE":
            first_val = df["RESULT TYPE"].iloc[0].strip()
            if "Stream" in first_val:
                stream_parts.append(df)
            elif "Rr" in first_val or "RR" in first_val:
                if "Latency" in first_val:
                    latency_parts.append(df)
                else:
                    rr_parts.append(df)
        elif first_col == "TYPE":
            reliability_parts.append(df)

    summary_frames = []
    for df in stream_parts + rr_parts:
        summary_frames.append(_build_summary_from_table(df))

    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()

    if latency_parts and not summary_df.empty:
        latency_df = pd.concat(latency_parts, ignore_index=True)
        merge_keys = ["Driver", "Profile", "Parallelism", "Host Network",
                       "VM mode", "Service", "Message Size"]
        lat_mapped = pd.DataFrame()
        for old, new in COLUMN_MAP.items():
            if old in latency_df.columns:
                lat_mapped[new] = latency_df[old]
        if "AVG 99%TILE VALUE" in latency_df.columns:
            lat_mapped["99%tile Observed Latency"] = latency_df["AVG 99%TILE VALUE"].apply(
                parse_value_with_units
            )
        available_keys = [k for k in merge_keys if k in summary_df.columns and k in lat_mapped.columns]
        if available_keys and "99%tile Observed Latency" in lat_mapped.columns:
            summary_df = summary_df.merge(
                lat_mapped[available_keys + ["99%tile Observed Latency"]],
                on=available_keys,
                how="left",
            )

    reliability_frames = []
    for df in reliability_parts:
        mapped = pd.DataFrame()
        for old, new in COLUMN_MAP.items():
            if old in df.columns:
                mapped[new] = df[old]
        if "TYPE" in df.columns:
            mapped["Reliability Type"] = df["TYPE"]
        if "AVG VALUE" in df.columns:
            mapped["Value"] = df["AVG VALUE"].apply(parse_value_with_units)
        reliability_frames.append(mapped)

    reliability_df = pd.concat(reliability_frames, ignore_index=True) if reliability_frames else pd.DataFrame()

    for df in [summary_df, reliability_df]:
        if df.empty:
            continue
        for c in BOOL_COLS:
            if c in df.columns:
                df[c] = df[c].apply(normalize_bool)
        if "Message Size" in df.columns:
            df["Message Size"] = pd.to_numeric(df["Message Size"], errors="coerce")
        if "Parallelism" in df.columns:
            df["Parallelism"] = pd.to_numeric(df["Parallelism"], errors="coerce")

    if not summary_df.empty:
        summary_df["Avg Throughput"] = pd.to_numeric(summary_df["Avg Throughput"], errors="coerce")
        for col in ["Confidence metric - low", "Confidence metric - high", "99%tile Observed Latency"]:
            if col in summary_df.columns:
                summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")

    if not reliability_df.empty and "Value" in reliability_df.columns:
        reliability_df["Value"] = pd.to_numeric(reliability_df["Value"], errors="coerce")

    node_cpu_df, pod_cpu_df, pod_mem_df = _extract_csv_sections(text)
    return summary_df, reliability_df, node_cpu_df, pod_cpu_df, pod_mem_df


def _extract_csv_sections(text):
    """Extract CSV sections embedded in a build log (node CPU and pod CPU)."""
    lines = text.splitlines()
    node_sections = []
    pod_sections = []
    current = []
    in_csv = False
    csv_type = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Role,"):
            if current:
                if csv_type == "node":
                    node_sections.append("\n".join(current))
                elif csv_type == "pod":
                    pod_sections.append("\n".join(current))
            current = [stripped]
            in_csv = True
            if "Idle CPU" in stripped:
                csv_type = "node"
            elif "Pod Name" in stripped and "Utilization" in stripped:
                csv_type = "pod"
            else:
                csv_type = None
                in_csv = False
                current = []
        elif in_csv:
            if stripped and "," in stripped and not stripped.startswith("+") and not stripped.startswith("|") and not stripped.startswith("time="):
                first = stripped.split(",", 1)[0].strip()
                if first in ("Client", "Server"):
                    current.append(stripped)
                else:
                    in_csv = False
                    if current and csv_type:
                        if csv_type == "node":
                            node_sections.append("\n".join(current))
                        elif csv_type == "pod":
                            pod_sections.append("\n".join(current))
                    current = []
                    csv_type = None
            else:
                in_csv = False
                if current and csv_type:
                    if csv_type == "node":
                        node_sections.append("\n".join(current))
                    elif csv_type == "pod":
                        pod_sections.append("\n".join(current))
                current = []
                csv_type = None

    if current and csv_type:
        if csv_type == "node":
            node_sections.append("\n".join(current))
        elif csv_type == "pod":
            pod_sections.append("\n".join(current))

    node_cpu_df = pd.DataFrame()
    for section in node_sections:
        df = parse_node_cpu(section)
        if not df.empty:
            node_cpu_df = pd.concat([node_cpu_df, df], ignore_index=True) if not node_cpu_df.empty else df

    pod_cpu_df = pd.DataFrame()
    pod_mem_df = pd.DataFrame()
    for section in pod_sections:
        df = pd.read_csv(StringIO(section))
        for c in BOOL_COLS:
            if c in df.columns:
                df[c] = df[c].apply(normalize_bool)
        df["Message Size"] = pd.to_numeric(df["Message Size"], errors="coerce")
        df["Parallelism"] = pd.to_numeric(df["Parallelism"], errors="coerce")
        df["Utilization"] = pd.to_numeric(df["Utilization"], errors="coerce")
        first_util = df["Utilization"].dropna().iloc[0] if not df["Utilization"].dropna().empty else 0
        if first_util > 10000:
            df["RSS (MB)"] = df["Utilization"] / (1024 * 1024)
            pod_mem_df = pd.concat([pod_mem_df, df], ignore_index=True) if not pod_mem_df.empty else df
        else:
            df["Utilization (%)"] = df["Utilization"] / 10
            pod_cpu_df = pd.concat([pod_cpu_df, df], ignore_index=True) if not pod_cpu_df.empty else df

    return node_cpu_df, pod_cpu_df, pod_mem_df


# ============================================================
# LOAD DATA
# ============================================================

PLOT_TEMPLATES = ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"]
selected_template = st.sidebar.selectbox("Plot theme", PLOT_TEMPLATES, index=0)
pio.templates.default = selected_template

build_log_url = st.sidebar.text_input(
    "Build log URL",
    placeholder="https://gcsweb-ci.apps.ci.l2s4.p1.openshiftapps.com/.../build-log.txt",
)
uploaded = st.sidebar.file_uploader("Or upload CSV / build log", type=["csv", "txt"])

if build_log_url:
    try:
        resp = requests.get(build_log_url, timeout=60)
        resp.raise_for_status()
        summary_df, reliability_df, node_cpu_df, pod_cpu_df, pod_mem_df = parse_text(resp.text)
    except requests.RequestException as e:
        st.error(f"Failed to fetch build log: {e}")
        st.stop()
elif uploaded is not None:
    raw_text = uploaded.getvalue().decode("utf-8")
    summary_df, reliability_df, node_cpu_df, pod_cpu_df, pod_mem_df = parse_text(raw_text)
else:
    st.info("Paste a build log URL or upload a CSV / build log file to get started.")
    st.stop()

if summary_df.empty:
    st.error("Could not parse summary data from the CSV.")
    st.stop()

stream_df = summary_df[summary_df["Profile"].isin(["TCP_STREAM", "UDP_STREAM"])].copy()
rr_df = summary_df[summary_df["Profile"].isin(["TCP_RR", "TCP_CRR"])].copy()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Filters")

all_profiles = sorted(summary_df["Profile"].dropna().unique())
selected_profiles = st.sidebar.multiselect(
    "Profile", options=all_profiles, default=all_profiles
)

parallelism_options = sorted(summary_df["Parallelism"].dropna().unique())
selected_parallelism = st.sidebar.multiselect(
    "Parallelism", options=parallelism_options, default=parallelism_options
)

host_network_filter = st.sidebar.multiselect(
    "Host Network", options=[True, False], default=[True, False]
)

virt_mode_filter = st.sidebar.multiselect(
    "VM Mode", options=[True, False], default=[True, False]
)

service_filter = st.sidebar.multiselect(
    "Service", options=[True, False], default=[True, False]
)


def apply_filters(df):
    if df.empty:
        return df
    if "Profile" in df.columns:
        df = df[df["Profile"].isin(selected_profiles)]
    if "Parallelism" in df.columns:
        df = df[df["Parallelism"].isin(selected_parallelism)]
    if "Host Network" in df.columns:
        df = df[df["Host Network"].isin(host_network_filter)]
    if "VM mode" in df.columns:
        df = df[df["VM mode"].isin(virt_mode_filter)]
    if "Service" in df.columns:
        df = df[df["Service"].isin(service_filter)]
    return df


stream_df = apply_filters(stream_df)
rr_df = apply_filters(rr_df)
reliability_df = apply_filters(reliability_df)
node_cpu_df = apply_filters(node_cpu_df)
pod_cpu_df = apply_filters(pod_cpu_df)
pod_mem_df = apply_filters(pod_mem_df)


# ============================================================
# KPI SECTION
# ============================================================

#st.header("Executive Summary")
#col1, col2, col3, col4 = st.columns(4)

#tcp_stream = stream_df[stream_df["Profile"] == "TCP_STREAM"] if not stream_df.empty else pd.DataFrame()
#udp_stream = stream_df[stream_df["Profile"] == "UDP_STREAM"] if not stream_df.empty else pd.DataFrame()
#tcp_rr = rr_df[rr_df["Profile"] == "TCP_RR"] if not rr_df.empty else pd.DataFrame()

#if not tcp_stream.empty:
#    col1.metric("Best TCP Throughput", f"{tcp_stream['Avg Throughput'].max():,.0f} Mb/s")

#if not udp_stream.empty:
#    col2.metric("Best UDP Throughput", f"{udp_stream['Avg Throughput'].max():,.0f} Mb/s")

#if not tcp_rr.empty:
#    col3.metric("Best TCP_RR", f"{tcp_rr['Avg Throughput'].max():,.0f} OP/s")

#if not summary_df.empty and "99%tile Observed Latency" in summary_df.columns:
#    latency_vals = apply_filters(summary_df)["99%tile Observed Latency"].dropna()
#    if not latency_vals.empty:
#        col4.metric("Lowest P99 Latency", f"{latency_vals.min():.1f} usec")


def sort_configs_by_base(columns):
    """Sort config columns so p=1 and p=2 of the same base config are adjacent."""
    def sort_key(col):
        base = re.sub(r"\s*p=\d+", "", col).strip()
        p_match = re.search(r"p=(\d+)", col)
        p_val = int(p_match.group(1)) if p_match else 0
        return (base, p_val)
    return sorted(columns, key=sort_key)


def bump_axis_fonts(fig, tick_size=14, title_size=14, legend_size=16, annotation_size=16):
    fig.update_xaxes(tickfont=dict(size=tick_size), title_font=dict(size=title_size))
    fig.update_yaxes(tickfont=dict(size=tick_size), title_font=dict(size=title_size))
    fig.update_annotations(font_size=annotation_size)
    fig.update_layout(legend=dict(font=dict(size=legend_size)))
    return fig


# ============================================================
# THROUGHPUT SECTION
# ============================================================

st.header("Stream: Throughput Analysis")

if not stream_df.empty:
    tcp_tp_tab, udp_tp_tab = st.tabs(["TCP_STREAM", "UDP_STREAM"])

    for tp_tab, protocol in [(tcp_tp_tab, "TCP_STREAM"), (udp_tp_tab, "UDP_STREAM")]:
        with tp_tab:
            chart_df = stream_df[stream_df["Profile"] == protocol].copy()

            if chart_df.empty:
                st.info(f"No data for {protocol}.")
                continue

            chart_df["config"] = (
                "host=" + chart_df["Host Network"].astype(str)
                + " vm=" + chart_df["VM mode"].astype(str)
                + " svc=" + chart_df["Service"].astype(str)
            )

            has_ci = ("Confidence metric - low" in chart_df.columns
                      and "Confidence metric - high" in chart_df.columns)

            fig = go.Figure()
            colors = px.colors.qualitative.Plotly
            config_colors = {cfg: colors[i % len(colors)] for i, cfg in enumerate(sorted(chart_df["config"].unique()))}

            for config in sorted(chart_df["config"].unique()):
                for par in sorted(chart_df["Parallelism"].unique()):
                    subset = chart_df[(chart_df["config"] == config) & (chart_df["Parallelism"] == par)]
                    if subset.empty:
                        continue
                    par_int = int(par)
                    dash = "dash" if par_int == 2 else "solid"
                    label = f"{config} p={par_int}" + (" (dashed)" if par_int == 2 else "")

                    error_y = None
                    if has_ci:
                        error_y = dict(
                            type="data",
                            symmetric=False,
                            array=subset["Confidence metric - high"] - subset["Avg Throughput"],
                            arrayminus=subset["Avg Throughput"] - subset["Confidence metric - low"],
                        )

                    fig.add_trace(go.Scatter(
                        x=subset["Message Size"],
                        y=subset["Avg Throughput"],
                        mode="lines+markers",
                        name=label,
                        line=dict(dash=dash, color=config_colors[config]),
                        marker=dict(color=config_colors[config]),
                        error_y=error_y,
                    ))

            fig.update_layout(
                title=f"{protocol} — Throughput vs Message Size (Confidence Intervals: P95)",
                xaxis_title="Message Size (bytes)",
                yaxis_title="Throughput (Mb/s)",
                height=600,
            )
            bump_axis_fonts(fig)
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# STREAM LOSS INVESTIGATION
# ============================================================

st.header("Stream: Loss Investigation")

if not reliability_df.empty:
    loss_tab_udp, loss_tab_tcp = st.tabs(["UDP Packet Loss", "TCP Retransmissions"])

    with loss_tab_udp:
        udp_loss = reliability_df[
            reliability_df["Reliability Type"].isin(["UDP Loss Percent", "UDP Percent Loss"])
        ].copy()

        if not udp_loss.empty:
            udp_loss["config"] = (
                "host=" + udp_loss["Host Network"].astype(str)
                + " vm=" + udp_loss["VM mode"].astype(str)
                + " svc=" + udp_loss["Service"].astype(str)
                + " p=" + udp_loss["Parallelism"].astype(int).astype(str)
            )

            udp_loss["Message Size"] = udp_loss["Message Size"].astype(int).astype(str)
            pivot = udp_loss.pivot_table(
                index="Message Size",
                columns="config",
                values="Value",
                aggfunc="mean",
            )
            size_order = sorted(pivot.index, key=lambda x: int(x))
            pivot = pivot.loc[size_order, sort_configs_by_base(pivot.columns)]

            text_matrix = pivot.copy()
            for col in text_matrix.columns:
                if "p=2" in col:
                    text_matrix[col] = text_matrix[col].apply(lambda v: f"<b>{v:.2f} ◆</b>" if pd.notna(v) else "")
                else:
                    text_matrix[col] = text_matrix[col].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "")

            fig = px.imshow(
                pivot,
                aspect="auto",
                title="UDP Packet Loss %<br><sup>◆ = p=2</sup>",
                labels=dict(x="Configuration", y="Message Size (bytes)", color="Loss %"),
            )
            fig.update_traces(text=text_matrix.values, texttemplate="%{text}")
            fig.update_layout(height=700)
            bump_axis_fonts(fig)
            st.plotly_chart(fig, use_container_width=True)

    with loss_tab_tcp:
        retrans = reliability_df[
            reliability_df["Reliability Type"] == "TCP Retransmissions"
        ].copy()

        if not retrans.empty:
            retrans["config"] = (
                "host=" + retrans["Host Network"].astype(str)
                + " vm=" + retrans["VM mode"].astype(str)
                + " svc=" + retrans["Service"].astype(str)
                + " p=" + retrans["Parallelism"].astype(int).astype(str)
            )

            retrans["Message Size"] = retrans["Message Size"].astype(int).astype(str)
            pivot = retrans.pivot_table(
                index="Message Size",
                columns="config",
                values="Value",
                aggfunc="mean",
            )
            size_order = sorted(pivot.index, key=lambda x: int(x))
            pivot = pivot.loc[size_order, sort_configs_by_base(pivot.columns)]

            text_matrix = pivot.copy()
            for col in text_matrix.columns:
                if "p=2" in col:
                    text_matrix[col] = text_matrix[col].apply(lambda v: f"<b>{v:.0f} ◆</b>" if pd.notna(v) else "")
                else:
                    text_matrix[col] = text_matrix[col].apply(lambda v: f"{v:.0f}" if pd.notna(v) else "")

            fig = px.imshow(
                pivot,
                aspect="auto",
                title="TCP Retransmissions<br><sup>◆ = p=2</sup>",
                labels=dict(x="Configuration", y="Message Size (bytes)", color="Retransmissions"),
            )
            fig.update_traces(text=text_matrix.values, texttemplate="%{text}")
            fig.update_layout(height=700)
            bump_axis_fonts(fig)
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# REQUEST-RESPONSE TRANSACTIONS
# ============================================================

st.header("Request-Response: Transactions")

if not rr_df.empty and "Avg Throughput" in rr_df.columns:
    rr_tx_df = rr_df[rr_df["Avg Throughput"].notna()].copy()

    if not rr_tx_df.empty:
        rr_tx_df["config"] = (
            "host=" + rr_tx_df["Host Network"].astype(str)
            + " vm=" + rr_tx_df["VM mode"].astype(str)
            + " svc=" + rr_tx_df["Service"].astype(str)
        )

        has_ci = ("Confidence metric - low" in rr_tx_df.columns
                  and "Confidence metric - high" in rr_tx_df.columns)

        fig = go.Figure()
        for profile in sorted(rr_tx_df["Profile"].unique()):
            pdf = rr_tx_df[rr_tx_df["Profile"] == profile]
            for par in sorted(pdf["Parallelism"].unique()):
                subset = pdf[pdf["Parallelism"] == par]
                par_int = int(par)
                pattern = "/" if par_int == 2 else ""
                label = f"{profile} p={par_int}"

                error_y = None
                if has_ci:
                    error_y = dict(
                        type="data",
                        symmetric=False,
                        array=subset["Confidence metric - high"] - subset["Avg Throughput"],
                        arrayminus=subset["Avg Throughput"] - subset["Confidence metric - low"],
                    )

                fig.add_trace(go.Bar(
                    x=subset["config"],
                    y=subset["Avg Throughput"],
                    name=label + (" (hatched)" if par_int == 2 else ""),
                    error_y=error_y,
                    marker_pattern_shape=pattern,
                ))

        fig.update_layout(
            barmode="group",
            title="TCP_RR and TCP_CRR — Avg Transactions, MSS=1024 (Confidence Intervals: P95)<br>",
            xaxis_title="Configuration",
            yaxis_title="Transactions (OP/s)",
            height=600,
        )
        bump_axis_fonts(fig)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# REQUEST-RESPONSE P99 LATENCY
# ============================================================

st.header("Request-Response: Latency")

if not rr_df.empty and "99%tile Observed Latency" in rr_df.columns:
    rr_lat_df = rr_df[rr_df["99%tile Observed Latency"].notna()].copy()

    if not rr_lat_df.empty:
        rr_lat_df["config"] = (
            "host=" + rr_lat_df["Host Network"].astype(str)
            + " vm=" + rr_lat_df["VM mode"].astype(str)
            + " svc=" + rr_lat_df["Service"].astype(str)
        )
        rr_lat_df["Bar Style"] = rr_lat_df["Parallelism"].astype(int).map(
            {1: "p=1", 2: "p=2 (hatched)"}
        ).fillna("p=" + rr_lat_df["Parallelism"].astype(int).astype(str))

        fig = go.Figure()
        profile_colors = {"TCP_CRR": "#636EFA", "TCP_RR": "#EF553B"}
        for profile in sorted(rr_lat_df["Profile"].unique()):
            pdf = rr_lat_df[rr_lat_df["Profile"] == profile]
            color = profile_colors.get(profile, "#636EFA")
            for par in sorted(pdf["Parallelism"].unique()):
                subset = pdf[pdf["Parallelism"] == par]
                par_int = int(par)
                pattern = "/" if par_int == 2 else ""
                bar_color = "#00CC96" if par_int == 2 else color
                label = f"{profile} p={par_int}" + (" (hatched)" if par_int == 2 else "")

                fig.add_trace(go.Bar(
                    x=subset["config"],
                    y=subset["99%tile Observed Latency"],
                    name=label,
                    marker=dict(color=bar_color, pattern_shape=pattern),
                ))

        fig.update_layout(
            barmode="group",
            title="TCP_RR and TCP_CRR — Latency, MSS=1024 (Average P99)<br>",
            xaxis_title="Configuration",
            yaxis_title="P99 Latency (usec)",
            height=600,
        )
        bump_axis_fonts(fig)
        st.plotly_chart(fig, use_container_width=True)






# ============================================================
# NODE CPU BY MESSAGE SIZE
# ============================================================

st.header("Node CPU Utilization")

if not node_cpu_df.empty:
    cpu_focus_cols = ["Idle CPU", "User CPU", "System CPU", "SoftIRQ CPU", "IRQ CPU", "IOWait CPU", "Steal CPU"]

    available_node_profiles = sorted(node_cpu_df["Profile"].dropna().unique())
    node_cpu_tabs = st.tabs(available_node_profiles)

    for sys_tab, sys_profile in zip(node_cpu_tabs, available_node_profiles):
        with sys_tab:
            prof_df = node_cpu_df[node_cpu_df["Profile"] == sys_profile].copy()
            if prof_df.empty:
                st.info(f"No CPU data for {sys_profile}.")
                continue

            prof_df["config"] = (
                "host=" + prof_df["Host Network"].astype(str)
                + " vm=" + prof_df["VM mode"].astype(str)
                + " svc=" + prof_df["Service"].astype(str)
            )
            prof_df["config_p"] = prof_df["config"] + " p=" + prof_df["Parallelism"].astype(int).astype(str)
            prof_df["Bar Style"] = prof_df["Parallelism"].astype(int).map(
                {1: "p=1", 2: "p=2 (hatched)"}
            ).fillna("p=" + prof_df["Parallelism"].astype(int).astype(str))

            available = [c for c in cpu_focus_cols if c in prof_df.columns]

            melted = prof_df.melt(
                id_vars=["Message Size", "config", "config_p", "Role", "Bar Style"],
                value_vars=available,
                var_name="CPU Component",
                value_name="CPU %",
            )

            melted.sort_values(["config", "Bar Style"], inplace=True)

            fig = px.bar(
                melted,
                x="config_p",
                y="CPU %",
                color="CPU Component",
                pattern_shape="Bar Style",
                pattern_shape_map={"p=1": "", "p=2 (hatched)": "/"},
                facet_col="Message Size",
                facet_row="Role",
                barmode="stack",
                title=f"{sys_profile} — Node CPU Breakdown",
                category_orders={"config_p": sorted(
                    melted["config_p"].unique(),
                    key=lambda c: (re.sub(r"\s*p=\d+", "", c), int(re.search(r"p=(\d+)", c).group(1)) if re.search(r"p=(\d+)", c) else 0)
                )},
            )
            for trace in fig.data:
                if "Idle CPU" in trace.name:
                    trace.visible = "legendonly"
            fig.update_yaxes(title_text="", matches="y")
            fig.update_yaxes(title_text="CPU %", col=1)
            fig.update_xaxes(title_text="")
            n_cols = len(prof_df["Message Size"].dropna().unique())
            mid_col = (n_cols // 2) + 1
            fig.update_xaxes(title_text="Configuration", row=1, col=mid_col)
            fig.update_layout(height=700)
            bump_axis_fonts(fig)
            st.plotly_chart(fig, use_container_width=True)



# ============================================================
# POD CPU UTILIZATION
# ============================================================

st.header("Pod CPU Utilization")

if not pod_cpu_df.empty and "Pod Name" in pod_cpu_df.columns:
    available_pod_profiles = sorted(pod_cpu_df["Profile"].dropna().unique())
    pod_tabs = st.tabs(available_pod_profiles)

    for pod_tab, pod_profile in zip(pod_tabs, available_pod_profiles):
        with pod_tab:
            prof_pod = pod_cpu_df[pod_cpu_df["Profile"] == pod_profile].copy()
            if prof_pod.empty:
                st.info(f"No pod CPU data for {pod_profile}.")
                continue

            prof_pod["Pod"] = prof_pod["Pod Name"].apply(
                lambda n: re.sub(r"(-[a-z0-9]{4,}){1,2}$", "", n)
            )
            all_pods = sorted(prof_pod["Pod"].unique())
            with st.expander("Filter pods", expanded=False):
                selected_pods = st.multiselect(
                    "Pods to display", all_pods, default=all_pods, key=f"pod_select_{pod_profile}"
                )
            prof_pod = prof_pod[prof_pod["Pod"].isin(selected_pods)]

            prof_pod["config"] = (
                "host=" + prof_pod["Host Network"].astype(str)
                + " vm=" + prof_pod["VM mode"].astype(str)
                + " svc=" + prof_pod["Service"].astype(str)
            )
            prof_pod["Bar Style"] = prof_pod["Parallelism"].astype(int).map(
                {1: "p=1", 2: "p=2 (hatched)"}
            ).fillna("p=" + prof_pod["Parallelism"].astype(int).astype(str))
            prof_pod.sort_values(["Pod", "config", "Bar Style"], inplace=True)

            fig = px.bar(
                prof_pod,
                x="Pod",
                y="Utilization (%)",
                color="config",
                pattern_shape="Bar Style",
                pattern_shape_map={"p=1": "", "p=2 (hatched)": "/"},
                facet_row="Role",
                facet_col="Message Size",
                barmode="group",
                title=f"{pod_profile} — Pod CPU Utilization",
                category_orders={
                    "Bar Style": ["p=1", "p=2 (hatched)"],
                    "config": sorted(prof_pod["config"].unique()),
                },
            )
            fig.update_yaxes(title_text="", matches="y")
            fig.update_yaxes(title_text="CPU Utilization (%)", col=1)
            fig.update_xaxes(title_text="", tickangle=45)
            n_cols = len(prof_pod["Message Size"].dropna().unique())
            mid_col = (n_cols // 2) + 1
            fig.update_xaxes(title_text="Pod", row=1, col=mid_col)
            fig.update_layout(height=800)
            bump_axis_fonts(fig)
            st.plotly_chart(fig, use_container_width=True)



# ============================================================
# POD MEMORY RSS UTILIZATION
# ============================================================

st.header("Pod Memory RSS Utilization")

if not pod_mem_df.empty and "Pod Name" in pod_mem_df.columns:
    available_mem_profiles = sorted(pod_mem_df["Profile"].dropna().unique())
    mem_tabs = st.tabs(available_mem_profiles)

    for mem_tab, mem_profile in zip(mem_tabs, available_mem_profiles):
        with mem_tab:
            prof_mem = pod_mem_df[pod_mem_df["Profile"] == mem_profile].copy()
            if prof_mem.empty:
                st.info(f"No pod memory data for {mem_profile}.")
                continue

            prof_mem["Pod"] = prof_mem["Pod Name"].apply(
                lambda n: re.sub(r"(-[a-z0-9]{4,}){1,2}$", "", n)
            )
            all_mem_pods = sorted(prof_mem["Pod"].unique())
            with st.expander("Filter pods", expanded=False):
                selected_mem_pods = st.multiselect(
                    "Pods to display", all_mem_pods, default=all_mem_pods, key=f"mem_pod_select_{mem_profile}"
                )
            prof_mem = prof_mem[prof_mem["Pod"].isin(selected_mem_pods)]

            prof_mem["config"] = (
                "host=" + prof_mem["Host Network"].astype(str)
                + " vm=" + prof_mem["VM mode"].astype(str)
                + " svc=" + prof_mem["Service"].astype(str)
            )
            prof_mem["Bar Style"] = prof_mem["Parallelism"].astype(int).map(
                {1: "p=1", 2: "p=2 (hatched)"}
            ).fillna("p=" + prof_mem["Parallelism"].astype(int).astype(str))

            fig = px.bar(
                prof_mem,
                x="Pod",
                y="RSS (MB)",
                color="config",
                pattern_shape="Bar Style",
                pattern_shape_map={"p=1": "", "p=2 (hatched)": "/"},
                facet_row="Role",
                facet_col="Message Size",
                barmode="group",
                title=f"{mem_profile} — Pod Memory RSS",
            )
            fig.update_yaxes(title_text="", matches="y")
            fig.update_yaxes(title_text="RSS (MB)", col=1)
            fig.update_xaxes(title_text="", tickangle=45)
            n_cols = len(prof_mem["Message Size"].dropna().unique())
            mid_col = (n_cols // 2) + 1
            fig.update_xaxes(title_text="Pod", row=1, col=mid_col)
            fig.update_layout(height=800)
            bump_axis_fonts(fig)
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# RAW DATA
# ============================================================

st.header("Raw Data")

tabs = st.tabs(["Throughput Stream", "Throughput and Latency RR", "Reliability", "Node CPU", "Pod CPU", "Pod Memory"])

with tabs[0]:
    st.dataframe(stream_df, use_container_width=True)

with tabs[1]:
    st.dataframe(rr_df, use_container_width=True)

with tabs[2]:
    st.dataframe(reliability_df, use_container_width=True)

with tabs[3]:
    st.dataframe(node_cpu_df, use_container_width=True)

with tabs[4]:
    st.dataframe(pod_cpu_df, use_container_width=True)

with tabs[5]:
    st.dataframe(pod_mem_df, use_container_width=True)
