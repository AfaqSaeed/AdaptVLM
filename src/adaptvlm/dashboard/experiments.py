"""Model-free local dashboard for controlled benchmark reports."""

from pathlib import Path

from adaptvlm.benchmarking.reporting import make_figures, read_benchmark, summary_rows


def build_experiment_app(root: str | Path = "reports"):
    """Select suites under one report directory; show measurements, failures and provenance."""
    import gradio as gr

    root = Path(root).resolve()

    def choices():
        return sorted(
            p.relative_to(root).as_posix()
            for p in root.rglob("benchmark.json")
            if p.resolve().is_relative_to(root)
        )

    def refresh():
        available = choices()
        return gr.Dropdown(choices=available, value=available[0] if available else None)

    def show(selection):
        if not selection:
            raise gr.Error("Run a benchmark and select its report first")
        path = (root / selection).resolve()
        if not path.is_relative_to(root):
            raise gr.Error("Report must be within the selected reports directory")
        try:
            report = read_benchmark(path)
            figures = make_figures(report)
            notice = (
                "**Software-test data: not research evidence.**"
                if report["software_test_only"]
                else "Measured observations; no statistical significance or speedup claim."
            )
            if report["comparison_limitations"]:
                notice += "\n\n" + "\n".join(report["comparison_limitations"])
            metadata = {k: v for k, v in report.items() if k != "results"}
            metadata["variants"] = [
                {k: v for k, v in result.items() if k != "samples"} for result in report["results"]
            ]
            return notice, summary_rows(report), *figures.values(), metadata
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title="AdaptVLM efficiency experiments", analytics_enabled=False) as app:
        gr.Markdown(
            "# AdaptVLM · Efficiency experiments\nCompare recorded suites without loading a model."
        )
        picker = gr.Dropdown(choices=choices(), label="Benchmark report")
        gr.Button("Refresh reports").click(refresh, outputs=picker, api_name="refresh")
        button = gr.Button("Load experiment")
        notice = gr.Markdown()
        table = gr.JSON(label="Measured results and unsupported configurations")
        latency = gr.Plot(label="Latency versus quality")
        memory = gr.Plot(label="GPU memory versus quality")
        state = gr.Plot(label="Serialized state size versus quality")
        details = gr.JSON(label="Hardware, configurations, categories and measurement conditions")
        button.click(
            show, picker, [notice, table, latency, memory, state, details], api_name="experiment"
        )
    return app
