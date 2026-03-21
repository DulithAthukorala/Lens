from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

console = Console()


def _ensure_kb() -> None:
    """Auto-ingest case studies if the ChromaDB sections collection is missing."""
    import chromadb
    from pathlib import Path

    client = chromadb.PersistentClient(path="chroma_data")
    collections = [c.name for c in client.list_collections()]
    if "lens_case_studies_sections" not in collections:
        console.print("[yellow]ChromaDB collection missing — running KB ingestion...[/yellow]")
        from src.ingestion.kb_builder import ingest
        ingest("sections")
        ingest("sentences")
        ingest("fixed512")
        console.print("[green]KB ingestion complete.[/green]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LENS — Multi-agent RAG pitch pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python src/main.py --url https://example.com",
    )
    parser.add_argument("--url", required=True, help="Prospect business URL")
    args = parser.parse_args()

    url: str = args.url

    _ensure_kb()

    console.print(f"\n[bold cyan]LENS[/bold cyan] analyzing: [bold]{url}[/bold]\n")

    from src.graph import run_pipeline

    result = run_pipeline(url)

    ro = result.get("researcher_output")
    ao = result.get("analyst_output")
    wo = result.get("writer_output")
    vo = result.get("validator_output")

    # --- Signals table ---
    if ro and ro.signals:
        sig_table = Table(title="Detected Failure Signals", show_lines=True)
        sig_table.add_column("Tier", style="bold red", width=8)
        sig_table.add_column("Signal", style="bold")
        sig_table.add_column("Evidence", style="yellow")
        sig_table.add_column("Conf.", width=6)
        for s in sorted(ro.signals, key=lambda x: x.tier.value):
            sig_table.add_row(s.tier.value, s.signal_name, s.evidence_value, f"{s.confidence:.0%}")
        console.print(sig_table)
    else:
        console.print("[yellow]No failure signals detected.[/yellow]")

    # --- Case study matches ---
    if ao and ao.top_matches:
        console.print(f"\n[bold]Top case study match:[/bold] {ao.top_matches[0].case_study_id} "
                      f"(similarity: {ao.top_matches[0].similarity_score:.3f})")
        console.print(f"[dim]{ao.top_matches[0].match_rationale}[/dim]")

    # --- Pitch ---
    if wo:
        pitch_json = json.dumps(wo.pitch.model_dump(), indent=2)
        console.print(Panel(pitch_json, title="Generated Pitch", border_style="green"))

    # --- Quality gate ---
    if vo:
        status_color = "green" if vo.passed else "red"
        status_text = "PASS" if vo.passed else "FAIL"
        if vo.warning_flag:
            status_text += " ⚠ (max retries reached)"

        score_table = Table(title="Quality Gate", show_header=False, show_lines=False)
        score_table.add_column("Dimension", style="bold")
        score_table.add_column("Score")
        score_table.add_row("Specificity (40%)", f"{vo.specificity_score:.3f}")
        score_table.add_row("Alignment (40%)", f"{vo.alignment_score:.3f}")
        score_table.add_row("Comparability (20%)", f"{vo.comparability_score:.3f}")
        score_table.add_row("─" * 20, "─" * 8)
        score_table.add_row(f"[{status_color}]Composite[/{status_color}]",
                            f"[{status_color}]{vo.composite_score:.3f} — {status_text}[/{status_color}]")
        console.print(score_table)

        if vo.failure_reason:
            console.print(f"[red]Failure reason:[/red] {vo.failure_reason}")

    console.print()


if __name__ == "__main__":
    main()
