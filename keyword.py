import io
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

# ---------------------------------
# CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Ringover Transcripciones",
    page_icon="REDUCCION-AES-01.png",
    layout="wide"
)

CHECKPOINT_FILE = "ringover_checkpoint.csv"
AUTOSAVE_EVERY = 10


# ---------------------------------
# AUTH
# ---------------------------------
def check_password() -> bool:
    if st.session_state.get("authenticated", False):
        return True

    st.markdown("## Acceso restringido")
    st.image("REDUCCION-AES-01.png", width=180)
    password = st.text_input("Introduce la contraseña", type="password")
    login_button = st.button("Entrar", type="primary", use_container_width=True)

    if login_button:
        app_password = st.secrets.get("APP_PASSWORD", "")
        if password == app_password:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta")

    return False


if not check_password():
    st.stop()


# ---------------------------------
# CHECKPOINT
# ---------------------------------
def load_checkpoint() -> pd.DataFrame:
    if os.path.exists(CHECKPOINT_FILE):
        try:
            df = pd.read_csv(CHECKPOINT_FILE, dtype=str)
            return df.fillna("")
        except Exception as e:
            st.warning(f"No se pudo leer el checkpoint: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


def save_checkpoint(df: pd.DataFrame) -> None:
    df.to_csv(CHECKPOINT_FILE, index=False, encoding="utf-8-sig")


def clear_checkpoint() -> None:
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


# ---------------------------------
# HELPERS GENERALES
# ---------------------------------
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
    return str(value).strip().replace('"', "")


def extract_call_id_from_url(value: Any) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip().replace('"', "")
    match = re.search(r"call-logs/(\d+)", text)
    if match:
        return match.group(1)

    return ""


def make_headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": api_key.strip()}


def safe_json(response: requests.Response) -> Any:
    response.raise_for_status()
    return response.json()


def request_with_retry(
    url: str,
    headers: Dict[str, str],
    timeout: int = 30,
    retries: int = 3,
    retry_sleep: float = 1.5
) -> Any:
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            return safe_json(r)
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(retry_sleep)

    raise last_error


def get_call_data(call_id: str, headers: Dict[str, str], timeout: int = 30) -> Dict[str, Any]:
    url = f"https://public-api.ringover.com/v2/calls/{call_id}"
    return request_with_retry(url, headers=headers, timeout=timeout)


def get_transcription_data(call_id: str, headers: Dict[str, str], timeout: int = 30) -> Any:
    url = f"https://public-api.ringover.com/v2/transcriptions/{call_id}"
    return request_with_retry(url, headers=headers, timeout=timeout)


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


def prepare_source_df(source_df: pd.DataFrame) -> pd.DataFrame:
    work = source_df[["call_id", "agente"]].copy()
    work["call_id"] = work["call_id"].apply(clean_call_id)
    work["agente"] = work["agente"].fillna("").astype(str).str.strip()
    work = work[work["call_id"] != ""].copy()
    work = work.drop_duplicates(subset=["call_id", "agente"])
    return work


def init_session() -> None:
    if "results_df" not in st.session_state:
        st.session_state.results_df = load_checkpoint()
    if "prepared_df" not in st.session_state:
        st.session_state.prepared_df = pd.DataFrame()
    if "source_df_ready" not in st.session_state:
        st.session_state.source_df_ready = None


init_session()


# ---------------------------------
# UI
# ---------------------------------
st.title("Ringover · Preparación + Transcripciones")
st.caption(
    "Primero normaliza el fichero de actividades para obtener call_id y agente. "
    "Después descarga transcripciones y analiza keywords."
)

with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("API key de Ringover", type="password")
    keyword = st.text_input("Palabra o frase a buscar", value="Asegura facil")
    sleep_ms = st.number_input(
        "Pausa entre llamadas a la API (ms)",
        min_value=0,
        max_value=5000,
        value=200,
        step=50,
    )
    autosave_every = st.number_input(
        "Guardar checkpoint cada X llamadas",
        min_value=1,
        max_value=500,
        value=AUTOSAVE_EVERY,
        step=1,
    )
    only_success = st.checkbox("Ocultar filas con error", value=False)

    checkpoint_df_sidebar = load_checkpoint()
    if not checkpoint_df_sidebar.empty:
        st.success(f"Checkpoint detectado: {len(checkpoint_df_sidebar)} llamadas guardadas")
    else:
        st.info("No hay checkpoint guardado")


tab1, tab2 = st.tabs(["1. Preparar archivo de actividades", "2. Descargar transcripciones"])


# ---------------------------------
# TAB 1 - PREPARACIÓN
# ---------------------------------
with tab1:
    st.subheader("Preparar archivo de actividades")
    st.write(
        "Sube el Excel o CSV exportado de actividades para extraer automáticamente "
        "los call_id desde la columna con la URL."
    )

    activities_file = st.file_uploader(
        "Sube el fichero de actividades",
        type=["xlsx", "xls", "csv"],
        key="activities_uploader",
    )

    if activities_file is not None:
        file_name = activities_file.name.lower()

        try:
            if file_name.endswith(".csv"):
                raw_df = pd.read_csv(activities_file, dtype=str)
            else:
                raw_df = pd.read_excel(activities_file, dtype=str)
        except Exception as e:
            st.error(f"No se pudo leer el fichero: {e}")
            st.stop()

        raw_df.columns = [str(c).strip() for c in raw_df.columns]

        st.markdown("### Vista previa del fichero original")
        st.dataframe(raw_df.head(10), use_container_width=True)

        columns = list(raw_df.columns)

        default_url_idx = columns.index("I") if "I" in columns else min(8, len(columns) - 1)
        url_col = st.selectbox(
            "Columna que contiene la URL de la llamada",
            options=columns,
            index=default_url_idx if len(columns) > 0 else 0,
        )

        agent_candidates = [
            c for c in columns
            if "agent" in c.lower()
            or "owner" in c.lower()
            or "agente" in c.lower()
            or "user" in c.lower()
        ]
        default_agent = agent_candidates[0] if agent_candidates else columns[0]

        agent_col = st.selectbox(
            "Columna del agente",
            options=columns,
            index=columns.index(default_agent) if default_agent in columns else 0,
        )

        prepare_button = st.button("Preparar fichero", type="primary", use_container_width=True)

        if prepare_button:
            prepared_df = raw_df[[url_col, agent_col]].copy()
            prepared_df = prepared_df.rename(columns={url_col: "url_llamada", agent_col: "agente"})
            prepared_df["call_id"] = prepared_df["url_llamada"].apply(extract_call_id_from_url)
            prepared_df["agente"] = prepared_df["agente"].fillna("").astype(str).str.strip()

            prepared_df = prepared_df[prepared_df["call_id"] != ""].copy()
            prepared_df = prepared_df[["call_id", "agente", "url_llamada"]]
            prepared_df = prepared_df.drop_duplicates(subset=["call_id", "agente"])

            st.session_state.prepared_df = prepared_df

        if not st.session_state.prepared_df.empty:
            prepared_df = st.session_state.prepared_df.copy()

            total_filas = len(raw_df)
            total_validas = len(prepared_df)
            total_sin_call_id = total_filas - total_validas

            c1, c2, c3 = st.columns(3)
            c1.metric("Filas originales", total_filas)
            c2.metric("Call ID extraídos", total_validas)
            c3.metric("Filas sin call_id", total_sin_call_id)

            st.markdown("### Fichero preparado")
            st.dataframe(prepared_df, use_container_width=True)

            csv_prepared = io.StringIO()
            prepared_df[["call_id", "agente"]].to_csv(csv_prepared, index=False, encoding="utf-8-sig")

            st.download_button(
                "Descargar CSV preparado",
                data=csv_prepared.getvalue(),
                file_name="call_ids_por_agente.csv",
                mime="text/csv",
            )

            st.success("Ya tienes el fichero preparado. Puedes ir a la pestaña 2 directamente.")


# ---------------------------------
# TAB 2 - TRANSCRIPCIONES
# ---------------------------------
with tab2:
    st.subheader("Descargar transcripciones")

    uploaded_file = st.file_uploader(
        "Sube el CSV con columnas call_id y agente, o usa el preparado en la pestaña 1",
        type=["csv"],
        key="transcriptions_uploader",
    )

    source_df = None

    if not st.session_state.prepared_df.empty:
        st.info("Se ha detectado un fichero preparado en la pestaña 1. Puedes usarlo directamente.")
        if st.button("Usar fichero preparado", use_container_width=True):
            source_df = st.session_state.prepared_df[["call_id", "agente"]].copy()
            st.session_state["source_df_ready"] = source_df

    if st.session_state.get("source_df_ready") is not None:
        source_df = st.session_state["source_df_ready"]

    if uploaded_file is not None:
        try:
            source_df = pd.read_csv(uploaded_file, dtype=str)
        except Exception:
            source_df = pd.read_csv(uploaded_file, sep="\t", dtype=str)

    if source_df is not None:
        source_df.columns = [c.strip() for c in source_df.columns]
        st.markdown("### Vista previa del fichero a procesar")
        st.dataframe(source_df.head(10), use_container_width=True)

        missing = [c for c in ["call_id", "agente"] if c not in source_df.columns]
        if missing:
            st.error(f"Faltan columnas obligatorias: {missing}")
            st.stop()

        work_preview = prepare_source_df(source_df)
        checkpoint_df = load_checkpoint()

        processed_ids_preview = set()
        if not checkpoint_df.empty and "call_id" in checkpoint_df.columns:
            processed_ids_preview = set(checkpoint_df["call_id"].astype(str).tolist())

        pending_preview = work_preview[~work_preview["call_id"].astype(str).isin(processed_ids_preview)].copy()

        p1, p2, p3 = st.columns(3)
        p1.metric("Total a procesar", len(work_preview))
        p2.metric("Ya guardadas en checkpoint", len(processed_ids_preview))
        p3.metric("Pendientes", len(pending_preview))

        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            run_button = st.button("Descargar y analizar", type="primary", use_container_width=True)
        with col_b:
            reload_button = st.button("Recargar checkpoint", use_container_width=True)
        with col_c:
            clear_button = st.button("Limpiar resultados y checkpoint", use_container_width=True)

        if reload_button:
            st.session_state.results_df = load_checkpoint()
            st.rerun()

        if clear_button:
            st.session_state.results_df = pd.DataFrame()
            clear_checkpoint()
            st.success("Resultados y checkpoint eliminados.")
            st.rerun()

        if run_button:
            if not api_key.strip():
                st.error("Introduce la API key de Ringover.")
                st.stop()

            work = prepare_source_df(source_df)
            headers = make_headers(api_key)

            checkpoint_df = load_checkpoint()
            if checkpoint_df.empty:
                results_df = pd.DataFrame()
                processed_ids = set()
            else:
                results_df = checkpoint_df.copy()
                if "call_id" in checkpoint_df.columns:
                    processed_ids = set(checkpoint_df["call_id"].astype(str).tolist())
                else:
                    processed_ids = set()

            pending_work = work[~work["call_id"].astype(str).isin(processed_ids)].copy()

            total_all = len(work)
            total_done = len(processed_ids)
            total_pending = len(pending_work)

            st.info(
                f"Total llamadas: {total_all} · "
                f"Ya procesadas: {total_done} · "
                f"Pendientes: {total_pending}"
            )

            progress = st.progress(0)
            status = st.empty()

            if total_pending == 0:
                st.session_state.results_df = results_df
                status.success("No quedan llamadas pendientes. Ya estaba todo procesado.")
            else:
                for idx, row in enumerate(pending_work.itertuples(index=False), start=1):
                    call_id = row.call_id
                    agente = row.agente

                    status.info(
                        f"Procesando {idx}/{total_pending} · call_id={call_id} · agente={agente}"
                    )

                    result = fetch_one_call(call_id, agente, headers, sleep_ms=int(sleep_ms))
                    results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)

                    if idx % int(autosave_every) == 0 or idx == total_pending:
                        save_checkpoint(results_df)
                        st.session_state.results_df = results_df

                    progress.progress(idx / total_pending)

                save_checkpoint(results_df)
                st.session_state.results_df = results_df
                status.success("Proceso completado.")

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

            view_df["mentions_keyword"] = (
                view_df["text"]
                .fillna("")
                .astype(str)
                .apply(lambda x: count_mentions(x, keyword))
            )
            view_df["has_keyword_search"] = view_df["mentions_keyword"] > 0

            calls_with_keyword = int(view_df["has_keyword_search"].sum())
            calls_without_keyword = total_calls - calls_with_keyword

            pct_total = round((calls_with_keyword / total_calls) * 100, 2) if total_calls else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total llamadas", total_calls)
            m2.metric("Llamadas con keyword", calls_with_keyword)
            m3.metric("% con keyword", f"{pct_total}%")
            m4.metric("Llamadas sin keyword", calls_without_keyword)

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
                view_df[view_df["has_keyword_search"]][
                    [
                        c for c in [
                            "call_id",
                            "agente",
                            "start_time",
                            "duration",
                            "speaker",
                            "mentions_keyword",
                            "text",
                            "error",
                        ]
                        if c in view_df.columns or c == "mentions_keyword"
                    ]
                ],
                use_container_width=True,
            )

            st.markdown("### Detalle de llamadas sin coincidencia")
            st.dataframe(
                view_df[~view_df["has_keyword_search"]][
                    [
                        c for c in [
                            "call_id",
                            "agente",
                            "start_time",
                            "duration",
                            "speaker",
                            "mentions_keyword",
                            "text",
                            "error",
                        ]
                        if c in view_df.columns or c == "mentions_keyword"
                    ]
                ],
                use_container_width=True,
            )

            summary_without_df = (
                view_df.groupby("agente", dropna=False)
                .agg(
                    total_llamadas=("call_id", "count"),
                    llamadas_sin_keyword=("has_keyword_search", lambda x: (~x).sum()),
                )
                .reset_index()
            )

            summary_without_df["porcentaje_sin_keyword"] = (
                summary_without_df["llamadas_sin_keyword"]
                / summary_without_df["total_llamadas"]
                * 100
            ).round(2)

            st.markdown("### Resumen por agente · llamadas sin coincidencia")
            st.dataframe(summary_without_df, use_container_width=True)

            summary_without_csv = io.StringIO()
            summary_without_df.to_csv(summary_without_csv, index=False, encoding="utf-8-sig")
            st.download_button(
                "Descargar resumen sin keyword por agente",
                data=summary_without_csv.getvalue(),
                file_name="resumen_sin_keyword_por_agente.csv",
                mime="text/csv",
            )

        st.markdown("### Llamadas sin transcripción")
        sin_transcripcion_df = view_df[~view_df["tiene_transcripcion"]].copy()
        st.dataframe(
            sin_transcripcion_df[
                [
                    c for c in [
                        "call_id",
                        "agente",
                        "start_time",
                        "duration",
                        "speaker",
                        "error",
                    ]
                    if c in sin_transcripcion_df.columns
                ]
            ],
            use_container_width=True,
        )
