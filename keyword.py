import io
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Ringover Transcripciones", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def count_mentions(text: str, keyword: str) -> int:
    text_n = normalize_text(text)
    keyword_n = normalize_text(keyword)
    if not keyword_n:
        return 0
    return text_n.count(keyword_n)


def clean_call_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip().replace('"', "")
    return value


def make_headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": api_key.strip()}


def safe_json(response: requests.Response) -> Any:
    response.raise_for_status()
    return response.json()


def get_call_data(call_id: str, headers: Dict[str, str], timeout: int = 30) -> Dict[str, Any]:
    url = f"https://public-api.ringover.com/v2/calls/{call_id}"
    r = requests.get(url, headers=headers, timeout=timeout)
    return safe_json(r)


def get_transcription_data(call_id: str, headers: Dict[str, str], timeout: int = 30) -> Any:
    url = f"https://public-api.ringover.com/v2/transcriptions/{call_id}"
    r = requests.get(url, headers=headers, timeout=timeout)
    return safe_json(r)


def extract_start_time_and_duration(call_response: Dict[str, Any]) -> Tuple[Optional[str], Optional[Any]]:
    start_time = None
    duration = None

    if isinstance(call_response, dict):
        start_time = (
            call_response.get("start_time")
            or call_response.get("call", {}).get("start_time")
            or call_response.get("data", {}).get("start_time")
        )
        duration = (
            call_response.get("duration")
            or call_response.get("call", {}).get("duration")
            or call_response.get("data", {}).get("duration")
        )

    return start_time, duration


def extract_speeches(transcription_response: Any) -> List[Dict[str, Any]]:
    transcription_item = None

    if isinstance(transcription_response, list):
        if transcription_response:
            transcription_item = transcription_response[0]
    else:
        transcription_item = transcription_response

    if not transcription_item or not isinstance(transcription_item, dict):
        return []

    speeches = (
        transcription_item.get("transcription_data", {}).get("speeches")
        or transcription_item.get("speeches")
        or transcription_item.get("data", {}).get("transcription_data", {}).get("speeches")
        or []
    )

    if isinstance(speeches, list):
        return speeches
    return []


def build_text_and_speakers(speeches: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    if not speeches:
        return None, None

    def safe_start(item: Dict[str, Any]) -> float:
        try:
            return float(str(item.get("start", 0)).replace(",", "."))
        except Exception:
            return 0.0

    speeches_sorted = sorted(speeches, key=safe_start)

    texts: List[str] = []
    speakers: List[str] = []

    for speech in speeches_sorted:
        txt = speech.get("text")
        if txt and str(txt).strip():
            texts.append(str(txt).strip())

        speaker_id = speech.get("speaker_id")
        if speaker_id == 0:
            speakers.append("Cliente")
        else:
            speakers.append("Agente")

    text_final = " ".join(texts) if texts else None
    speaker_final = " / ".join(sorted(set(speakers))) if speakers else None
    return text_final, speaker_final


def fetch_one_call(call_id: str, agente: str, headers: Dict[str, str], sleep_ms: int = 200) -> Dict[str, Any]:
    start_time = None
    duration = None
    speaker = None
    text = None
    error_msg = None

    try:
        call_response = get_call_data(call_id, headers=headers)
        start_time, duration = extract_start_time_and_duration(call_response)
    except Exception as e:
        error_msg = f"Error call: {e}"

    try:
        transcription_response = get_transcription_data(call_id, headers=headers)
        speeches = extract_speeches(transcription_response)
        text, speaker = build_text_and_speakers(speeches)
    except Exception as e:
        if error_msg:
            error_msg += f" | Error transcription: {e}"
        else:
            error_msg = f"Error transcription: {e}"

    if sleep_ms > 0:
        time.sleep(sleep_ms / 1000)

    return {
        "call_id": call_id,
        "agente": agente,
        "start_time": start_time,
        "duration": duration,
        "speaker": speaker,
        "text": text,
        "error": error_msg,
    }


def analyze_keyword(df: pd.DataFrame, keyword: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["agente"] = work["agente"].fillna("SIN_AGENTE").astype(str)
    work["text"] = work["text"].fillna("").astype(str)
    work["mentions"] = work["text"].apply(lambda x: count_mentions(x, keyword))
    work["has_keyword"] = work["mentions"] > 0

    summary = (
        work.groupby("agente", dropna=False)
        .agg(
            total_llamadas=("call_id", "count"),
            llamadas_con_keyword=("has_keyword", "sum"),
            menciones_totales=("mentions", "sum"),
        )
        .reset_index()
    )

    summary["porcentaje_llamadas_con_keyword"] = (
        summary["llamadas_con_keyword"] / summary["total_llamadas"] * 100
    ).round(2)

    summary["media_menciones_por_llamada"] = (
        summary["menciones_totales"] / summary["total_llamadas"]
    ).round(2)

    summary = summary.sort_values(
        ["porcentaje_llamadas_con_keyword", "menciones_totales"],
        ascending=[False, False],
    )

    detail = work[work["has_keyword"]].copy()
    return summary, detail


results_df = st.session_state.results_df

if not results_df.empty:
    results_df = results_df.copy()
    results_df["tiene_transcripcion"] = results_df["text"].fillna("").astype(str).str.strip() != ""

    total_procesadas = len(results_df)
    total_con_transcripcion = int(results_df["tiene_transcripcion"].sum())
    total_sin_transcripcion = total_procesadas - total_con_transcripcion
    pct_con_transcripcion = round((total_con_transcripcion / total_procesadas) * 100, 2) if total_procesadas else 0

    st.subheader("Resumen de transcripciones")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Llamadas procesadas", total_procesadas)
    r2.metric("Con transcripción", total_con_transcripcion)
    r3.metric("Sin transcripción", total_sin_transcripcion)
    r4.metric("% con transcripción", f"{pct_con_transcripcion}%")

    view_df = results_df.copy()
    view_df["tiene_transcripcion"] = view_df["text"].fillna("").astype(str).str.strip() != ""

    if only_success:
        view_df = view_df[view_df["error"].isna() | (view_df["error"] == "")]

    st.subheader("Resultado de transcripciones")
    st.dataframe(view_df, use_container_width=True)

    csv_buffer = io.StringIO()
    view_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
    st.download_button(
        "Descargar CSV de transcripciones",
        data=csv_buffer.getvalue(),
        file_name="ringover_transcripciones_con_agente.csv",
        mime="text/csv",
    )

    if keyword.strip():
        summary_df, detail_df = analyze_keyword(view_df, keyword)

        st.subheader(f"Análisis de keyword: {keyword}")
        total_calls = len(view_df)
        calls_with_keyword = int((view_df["text"].fillna("").astype(str).apply(lambda x: count_mentions(x, keyword) > 0)).sum())
        pct_total = round((calls_with_keyword / total_calls) * 100, 2) if total_calls else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("Total llamadas", total_calls)
        m2.metric("Llamadas con keyword", calls_with_keyword)
        m3.metric("% sobre total", f"{pct_total}%")

        st.markdown("### Resumen por agente")
        st.dataframe(summary_df, use_container_width=True)

        summary_csv = io.StringIO()
        summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        st.download_button(
            "Descargar resumen por agente",
            data=summary_csv.getvalue(),
            file_name="resumen_keyword_por_agente.csv",
            mime="text/csv",
        )

        st.markdown("### Detalle de llamadas con coincidencia")
        st.dataframe(
            detail_df[[c for c in ["call_id", "agente", "start_time", "duration", "speaker", "mentions", "text", "error"] if c in detail_df.columns]],
            use_container_width=True,
        )
else:
    st.info("Sube un CSV, añade tu API key y pulsa ‘Descargar y analizar’.")
