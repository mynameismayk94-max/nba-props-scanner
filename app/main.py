        selected_date,
        market,
        limit
    )


@app.get("/capture-audit-snapshot", dependencies=[Depends(require_admin_api_key)])
def capture_audit_snapshot(
    selected_date: str = Query(alias="date"),
    market: str = Query(...)
):
    """Gera a população de auditoria sem registrar uso operacional do Scanner."""
    if market not in ACTIVE_MARKETS: