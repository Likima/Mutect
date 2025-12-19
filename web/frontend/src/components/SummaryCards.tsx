import React from "react";
import type { Summary } from "../types";
import { fetchSummary } from "../lib/api";

// SummaryCards component to display the summary of the predictions
export const SummaryCards: React.FC = () => {
	const [summary, setSummary] = React.useState<Summary | null>(null);
	const [error, setError] = React.useState<string | null>(null);

	// Fetch the summary of the predictions
	React.useEffect(() => {
		const ctrl = new AbortController();
		fetchSummary(ctrl.signal)
			.then(setSummary)
			.catch((e) => setError(String(e)));
		return () => ctrl.abort();
	}, []);

	// If there is an error, display it
	if (error) {
		return <div style={{ color: "tomato" }}>Failed to load summary: {error}</div>;
	}
	if (!summary) {
		return <div>Loading summary…</div>;
	}

	// Create the cards to display the summary of the predictions
	const cards = [
		{ label: "Total sequences", value: summary.total_sequences },
		{ label: "Predicted STRs", value: summary.predicted_strs },
		{ label: "Non-STRs", value: summary.predicted_non_strs },
		{ label: "Avg STR prob", value: summary.average_str_probability.toFixed(3) },
		{ label: "Threshold", value: summary.threshold }
	];

	// Return the section to display the summary of the predictions
	return (
		<section>
			<h2 style={{ fontSize: 16, margin: "0 0 8px 0" }}>Summary</h2>
			<div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(160px,1fr))", gap: 12 }}>
				{cards.map((c) => (
					<div key={c.label} style={{ border: "1px solid #444", borderRadius: 8, padding: 12 }}>
						<div style={{ fontSize: 12, opacity: 0.8 }}>{c.label}</div>
						<div style={{ fontSize: 22, fontWeight: 600, marginTop: 4 }}>{c.value}</div>
					</div>
				))}
			</div>

			<div style={{ marginTop: 12 }}>
				<h3 style={{ fontSize: 14, margin: "0 0 6px 0" }}>Top motifs</h3>
				<div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
					{Object.entries(summary.motif_statistics.most_common_motifs).map(([motif, count]) => (
						<span
							key={motif}
							style={{
								border: "1px solid #444",
								borderRadius: 999,
								padding: "6px 10px",
								fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace"
							}}
						>
							{motif} · {count}
						</span>
					))}
				</div>
			</div>
		</section>
	);
};


