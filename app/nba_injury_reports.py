"""Consulta conservadora dos relatórios oficiais de lesões da NBA.

A fonte só encerra uma pendência como DNP quando:
- o PDF oficial é baixado e validado;
- o confronto consta no relatório;
- o atleta é encontrado com status explícito "Out".

Ausência do atleta, status não reconhecido ou falha de download nunca viram DNP.
"""
from datetime import datetime, timedelta
from functools import lru_cache
from io import BytesIO
from zoneinfo import ZoneInfo
import re

import requests
from pypdf import PdfReader


NBA_INJURY_REPORT_BASE = (
    "https://ak-static.cms.nba.com/referee/injury/"
    "Injury-Report_{date}_{time}.pdf"
)
EASTERN = ZoneInfo("America/New_York")
SAO_PAULO = ZoneInfo("America/Sao_Paulo")
STATUS_PATTERN = re.compile(
    r"\b(Out|Doubtful|Questionable|Probable|Available)\b",
    re.IGNORECASE,
)


@lru_cache(maxsize=256)
def _fetch_report_text(url):
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 NBA Props Analytics"},
            timeout=5,
        )
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            return ""
        reader = PdfReader(BytesIO(response.content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        # Cacheia também URLs inexistentes/indisponíveis para evitar repetir
        # as mesmas tentativas em cada jogador e em cada mercado.
        return ""


def _player_name_patterns(player_name):
    name = re.sub(r"\s+", " ", (player_name or "").strip())
    if not name:
        return []

    parts = name.split(" ")
    patterns = [re.escape(name)]
    if len(parts) >= 2:
        patterns.append(
            re.escape(parts[-1]) + r"\s*,\s*" + re.escape(" ".join(parts[:-1]))
        )
    return patterns


def _status_for_player(report_text, matchup, player_name):
    normalized = re.sub(r"\s+", " ", report_text or "")
    if not normalized or (matchup and matchup.upper() not in normalized.upper()):
        return None

    for player_pattern in _player_name_patterns(player_name):
        for match in re.finditer(player_pattern, normalized, re.IGNORECASE):
            # A extração do PDF pode deslocar colunas. Só aceitamos status
            # explícito próximo ao nome; não inferimos pela ausência do atleta.
            nearby = normalized[match.end():match.end() + 140]
            status_match = STATUS_PATTERN.search(nearby)
            if status_match:
                return status_match.group(1).title()
    return None


def get_official_nba_injury_status(
    selected_date,
    start_time_brt,
    matchup,
    player_name,
):
    """Retorna status oficial pré-jogo; None significa evidência insuficiente."""
    if not all((selected_date, start_time_brt, matchup, player_name)):
        return None

    try:
        local_start = datetime.strptime(
            f"{selected_date} {start_time_brt}", "%Y-%m-%d %H:%M"
        ).replace(tzinfo=SAO_PAULO)
        tipoff_et = local_start.astimezone(EASTERN)
    except (TypeError, ValueError):
        return None

    # Temporada 2026-27: relatórios publicados em intervalos de 15 minutos.
    # Verificamos até duas horas antes do início, do mais recente ao mais antigo.
    for minutes_before in range(15, 121, 15):
        report_time = tipoff_et - timedelta(minutes=minutes_before)
        slot = report_time.strftime("%I_%M%p")
        report_date = report_time.strftime("%Y-%m-%d")
        url = NBA_INJURY_REPORT_BASE.format(date=report_date, time=slot)

        report_text = _fetch_report_text(url)
        if not report_text:
            continue

        status = _status_for_player(report_text, matchup, player_name)
        if status:
            return {
                "status": status,
                "source": "nba_official_injury_report",
                "report_url": url,
                "report_time_et": report_time.isoformat(),
            }

    return None
