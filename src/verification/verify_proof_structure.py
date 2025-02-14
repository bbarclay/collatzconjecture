#!/usr/bin/env python3
"""
Proof structure and dependency verification script.
"""

import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt


class ProofVerifier:
    """Verifies proof structure and dependencies"""

    def __init__(self, latex_dir: Path):
        self.latex_dir = latex_dir
        self.theorems: Dict[str, Tuple[str, str, Set[str]]] = (
            {}
        )  # label -> (file, content, dependencies)
        self.lemmas: Dict[str, Tuple[str, str, Set[str]]] = (
            {}
        )  # label -> (file, content, dependencies)
        self.propositions: Dict[str, Tuple[str, str, Set[str]]] = (
            {}
        )  # label -> (file, content, dependencies)
        self.corollaries: Dict[str, Tuple[str, str, Set[str]]] = (
            {}
        )  # label -> (file, content, dependencies)
        self.definitions: Dict[str, Tuple[str, str]] = {}  # label -> (file, content)
        self.errors: List[str] = []
        self.graph = nx.DiGraph()

    def scan_files(self):
        """Scan all LaTeX files for proof components"""
        for tex_file in self.latex_dir.glob("**/*.tex"):
            self._scan_file(tex_file)

    def _scan_file(self, file_path: Path):
        """Scan a single LaTeX file for proof components"""
        content = file_path.read_text()
        rel_path = file_path.relative_to(self.latex_dir)

        # Find theorems, lemmas, propositions, and corollaries
        for env_type in ["theorem", "lemma", "proposition", "corollary"]:
            pattern = (
                rf"\\begin{{{env_type}}}.*?\\label{{([^}}]+)}}(.*?)\\end{{{env_type}}}"
            )
            for match in re.finditer(pattern, content, re.DOTALL):
                label = match.group(1)
                env_content = match.group(2)
                deps = self._extract_dependencies(env_content)

                target_dict = getattr(self, f"{env_type}s")
                if label in target_dict:
                    self.errors.append(
                        f"{env_type.capitalize()} '{label}' redefined in {rel_path} "
                        f"(originally defined in {target_dict[label][0]})"
                    )
                else:
                    target_dict[label] = (str(rel_path), env_content, deps)

        # Find definitions
        for match in re.finditer(
            r"\\begin{definition}.*?\\label{([^}]+)}(.*?)\\end{definition}",
            content,
            re.DOTALL,
        ):
            label = match.group(1)
            def_content = match.group(2)
            if label in self.definitions:
                self.errors.append(
                    f"Definition '{label}' redefined in {rel_path} "
                    f"(originally defined in {self.definitions[label][0]})"
                )
            else:
                self.definitions[label] = (str(rel_path), def_content)

    def _extract_dependencies(self, content: str) -> Set[str]:
        """Extract dependencies from content"""
        deps = set()
        # Find references to other theorems/lemmas/etc
        for match in re.finditer(r"\\ref{([^}]+)}", content):
            deps.add(match.group(1))
        return deps

    def build_dependency_graph(self):
        """Build dependency graph of proof components"""
        # Add nodes
        for label, (file, _, _) in self.theorems.items():
            self.graph.add_node(label, type="theorem", file=file)
        for label, (file, _, _) in self.lemmas.items():
            self.graph.add_node(label, type="lemma", file=file)
        for label, (file, _, _) in self.propositions.items():
            self.graph.add_node(label, type="proposition", file=file)
        for label, (file, _, _) in self.corollaries.items():
            self.graph.add_node(label, type="corollary", file=file)
        for label, (file, _) in self.definitions.items():
            self.graph.add_node(label, type="definition", file=file)

        # Add edges
        for component_dict in [
            self.theorems,
            self.lemmas,
            self.propositions,
            self.corollaries,
        ]:
            for label, (_, _, deps) in component_dict.items():
                for dep in deps:
                    if dep in self.graph:
                        self.graph.add_edge(label, dep)

    def verify_structure(self) -> bool:
        """Verify proof structure"""
        valid = True

        # Check for circular dependencies
        try:
            cycles = list(nx.simple_cycles(self.graph))
            if cycles:
                for cycle in cycles:
                    self.errors.append(
                        f"Circular dependency detected: {' -> '.join(cycle)}"
                    )
                valid = False
        except nx.NetworkXNoCycle:
            pass

        # Check for undefined references
        all_components = {
            **self.theorems,
            **self.lemmas,
            **self.propositions,
            **self.corollaries,
            **{
                label: (file, content, set())
                for label, (file, content) in self.definitions.items()
            },
        }

        for label, (file, _, deps) in all_components.items():
            for dep in deps:
                if dep not in all_components:
                    self.errors.append(
                        f"Reference to undefined component '{dep}' in {label} ({file})"
                    )
                    valid = False

        # Check for proper dependency ordering
        for label, (file, _, deps) in all_components.items():
            for dep in deps:
                if dep in all_components:
                    dep_file = all_components[dep][0]
                    if (
                        dep_file > file
                    ):  # Simple check if dependency appears after usage
                        self.errors.append(
                            f"Component '{label}' in {file} depends on '{dep}' which appears later in {dep_file}"
                        )
                        valid = False

        return valid

    def generate_visualization(self):
        """Generate visualization of proof structure"""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)

        # Draw nodes with different colors for different types
        colors = {
            "theorem": "lightblue",
            "lemma": "lightgreen",
            "proposition": "lightcoral",
            "corollary": "lightyellow",
            "definition": "lightgray",
        }

        for node_type in colors:
            nodes = [
                n for n, d in self.graph.nodes(data=True) if d["type"] == node_type
            ]
            nx.draw_networkx_nodes(
                self.graph,
                pos,
                nodelist=nodes,
                node_color=colors[node_type],
                node_size=1000,
                alpha=0.8,
            )

        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, edge_color="gray", arrows=True)

        # Draw labels
        nx.draw_networkx_labels(self.graph, pos)

        # Add legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                label=node_type.capitalize(),
                markersize=10,
            )
            for node_type, color in colors.items()
        ]
        plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))

        plt.title("Proof Structure Visualization")
        plt.axis("off")
        plt.tight_layout()

        # Save plot
        plt.savefig(self.latex_dir / "proof_structure.png", bbox_inches="tight")
        plt.close()

    def generate_report(self) -> str:
        """Generate verification report"""
        report = [
            "# Proof Structure Verification Report",
            "",
            "## Statistics",
            f"- Total theorems: {len(self.theorems)}",
            f"- Total lemmas: {len(self.lemmas)}",
            f"- Total propositions: {len(self.propositions)}",
            f"- Total corollaries: {len(self.corollaries)}",
            f"- Total definitions: {len(self.definitions)}",
            "",
        ]

        if self.errors:
            report.extend(
                [
                    "## Structural Issues",
                    "",
                    *[f"- {error}" for error in self.errors],
                    "",
                ]
            )
        else:
            report.extend(
                [
                    "## Status",
                    "",
                    "Proof structure is valid with no issues detected.",
                    "",
                ]
            )

        # Add dependency summary
        report.extend(
            [
                "## Dependency Summary",
                "",
            ]
        )

        all_components = {
            **self.theorems,
            **self.lemmas,
            **self.propositions,
            **self.corollaries,
        }

        for label, (file, _, deps) in sorted(all_components.items()):
            if deps:
                report.append(
                    f"- {label} ({file}) depends on: {', '.join(sorted(deps))}"
                )

        return "\n".join(report)


def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: verify_proof_structure.py <latex_dir>")
        sys.exit(1)

    latex_dir = Path(sys.argv[1])
    if not latex_dir.is_dir():
        print(f"LaTeX directory not found: {latex_dir}")
        sys.exit(1)

    verifier = ProofVerifier(latex_dir)
    verifier.scan_files()
    verifier.build_dependency_graph()

    # Verify proof structure
    structure_valid = verifier.verify_structure()

    # Generate visualization
    verifier.generate_visualization()

    # Generate and print report
    report = verifier.generate_report()
    print(report)

    # Save report
    report_file = latex_dir / "proof_structure_check.md"
    report_file.write_text(report)
    print(f"\nReport saved to: {report_file}")
    print(f"Visualization saved to: {latex_dir}/proof_structure.png")

    # Exit with appropriate status
    sys.exit(not structure_valid)


if __name__ == "__main__":
    main()
