#!/usr/bin/env python3
"""
Report generator for Collatz conjecture proof verification.
Generates detailed reports from verification results.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

from tests.config import TEST_DATA


class ReportGenerator:
    """Generates detailed reports from verification results"""

    def __init__(self, results_file: Path):
        self.results_file = results_file
        with open(results_file) as f:
            self.results = json.load(f)

        # Set up output directory
        self.output_dir = Path(TEST_DATA["results_dir"]) / "reports"
        self.output_dir.mkdir(exist_ok=True)

        # Set style for plots
        plt.style.use("seaborn")
        sns.set_palette("husl")

    def generate_verification_summary(self) -> str:
        """Generate text summary of verification results"""
        stats = self.results["statistics"]
        summary = [
            "# Collatz Conjecture Proof Verification Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Overall Statistics",
            f"- Total theorems: {stats['total_theorems']}",
            f"- Verified theorems: {stats['verified_theorems']}",
            f"- Failed theorems: {stats['failed_theorems']}",
            f"- Verification coverage: {stats['verification_coverage']*100:.1f}%",
            "",
            "## Component Status",
        ]

        # Add component details
        for component, results in self.results["verification_results"].items():
            summary.extend(
                [
                    f"### {component.title()}",
                    self._format_component_results(results),
                    "",
                ]
            )

        # Add errors if any
        if self.results["errors"]:
            summary.extend(
                ["## Errors", *[f"- {error}" for error in self.results["errors"]], ""]
            )

        return "\n".join(summary)

    def _format_component_results(self, results: Dict[str, Any]) -> str:
        """Format results for a single component"""
        lines = []
        for theorem, result in results.items():
            status = (
                "✓" if isinstance(result, dict) and not result.get("errors") else "✗"
            )
            lines.append(f"- [{status}] {theorem.replace('_', ' ').title()}")
            if isinstance(result, dict):
                if "violations" in result:
                    lines.append(f"  - Found {len(result['violations'])} violations")
                if "statistics" in result:
                    for key, value in result["statistics"].items():
                        lines.append(f"  - {key}: {value}")
        return "\n".join(lines)

    def generate_plots(self):
        """Generate visualization plots"""
        self._plot_verification_coverage()
        self._plot_theorem_dependencies()
        if "information_theory" in self.results["verification_results"]:
            self._plot_entropy_violations()

    def _plot_verification_coverage(self):
        """Plot verification coverage by component"""
        components = []
        coverage = []

        for component, results in self.results["verification_results"].items():
            total = len(results)
            verified = sum(
                1
                for r in results.values()
                if isinstance(r, dict) and not r.get("errors")
            )
            components.append(component.title())
            coverage.append(verified / total if total > 0 else 0)

        plt.figure(figsize=(10, 6))
        plt.bar(components, coverage)
        plt.title("Verification Coverage by Component")
        plt.ylabel("Coverage")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()

        plot_file = self.output_dir / "coverage_plot.png"
        plt.savefig(plot_file)
        plt.close()

    def _plot_theorem_dependencies(self):
        """Plot theorem dependency graph"""
        try:
            import networkx as nx

            G = nx.DiGraph()

            # Add nodes and edges from verification results
            for component, results in self.results["verification_results"].items():
                for theorem, result in results.items():
                    G.add_node(theorem)
                    if isinstance(result, dict) and "dependencies" in result:
                        for dep in result["dependencies"]:
                            G.add_edge(dep, theorem)

            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G)
            nx.draw(
                G,
                pos,
                with_labels=True,
                node_color="lightblue",
                node_size=2000,
                font_size=8,
                font_weight="bold",
            )
            plt.title("Theorem Dependency Graph")

            plot_file = self.output_dir / "dependencies_plot.png"
            plt.savefig(plot_file)
            plt.close()

        except ImportError:
            print("networkx not installed, skipping dependency plot")

    def _plot_entropy_violations(self):
        """Plot entropy bound violations"""
        info_theory = self.results["verification_results"]["information_theory"]
        if "violations" in info_theory:
            violations = info_theory["violations"]
            if violations:
                n_values = [v["n"] for v in violations]
                errors = [
                    abs(v["bounds"]["actual"] - v["bounds"]["theoretical"])
                    for v in violations
                ]

                plt.figure(figsize=(10, 6))
                plt.scatter(n_values, errors)
                plt.title("Entropy Bound Violations")
                plt.xlabel("n")
                plt.ylabel("Error")
                plt.yscale("log")
                plt.tight_layout()

                plot_file = self.output_dir / "entropy_violations_plot.png"
                plt.savefig(plot_file)
                plt.close()

    def generate_report(self) -> Path:
        """Generate complete verification report"""
        # Generate summary
        summary = self.generate_verification_summary()

        # Generate plots
        self.generate_plots()

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"verification_report_{timestamp}.md"

        with open(report_file, "w") as f:
            f.write(summary)

            # Add plot references
            f.write("\n## Visualizations\n\n")
            for plot in self.output_dir.glob("*.png"):
                if plot.stem.endswith(timestamp):
                    continue
                f.write(f"![{plot.stem}]({plot.name})\n\n")

        return report_file


def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: generate_report.py <results_file>")
        sys.exit(1)

    results_file = Path(sys.argv[1])
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        sys.exit(1)

    generator = ReportGenerator(results_file)
    report_file = generator.generate_report()
    print(f"Report generated: {report_file}")


if __name__ == "__main__":
    main()
