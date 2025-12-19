import React from "react";
import type { Locus } from "../types";
import { fetchLoci } from "../lib/api";

export const LociList: React.FC = () => {
	const [loci, setLoci] = React.useState<Locus[]>([]);
	const [error, setError] = React.useState<string | null>(null);
	const [loading, setLoading] = React.useState(false);

	React.useEffect(() => {
		const ctrl = new AbortController();
		setLoading(true);
		fetchLoci(ctrl.signal)
			.then((r) => setLoci(r.loci))
			.catch((e) => setError(String(e)))
			.finally(() => setLoading(false));
		return () => ctrl.abort();
	}, []);

	if (error) return <div style={{ color: "tomato" }}>Failed to load loci: {error}</div>;
	if (loading) return <div>Loading loci…</div>;
	if (!loci.length) return <div>No loci available.</div>;

	return (
		<div style={{ border: "1px solid #444", borderRadius: 8, overflow: "hidden" }}>
			<div style={{ overflowX: "auto" }}>
				<table style={{ borderCollapse: "collapse", width: "100%" }}>
					<thead>
						<tr>
							<th style={th}>Locus</th>
							<th style={th}>Chrom</th>
							<th style={th}>Start</th>
							<th style={th}>End</th>
							<th style={th}>Motif</th>
							<th style={th}>Count</th>
							<th style={th}>Prob</th>
							<th style={th}>Read</th>
						</tr>
					</thead>
					<tbody>
						{loci.slice(0, 200).map((l) => (
							<tr key={l.locusId}>
								<td style={td}>{l.locusId}</td>
								<td style={td}>{l.chrom}</td>
								<td style={td}>{l.start}</td>
								<td style={td}>{l.end}</td>
								<td style={td}><code>{l.repeatMotif ?? ""}</code></td>
								<td style={td}>{l.repeatCount ?? ""}</td>
								<td style={td}>{l.probability.toFixed(3)}</td>
								<td style={td} title={l.readName}>{l.readName}</td>
							</tr>
						))}
					</tbody>
				</table>
			</div>
			<div style={{ padding: 12, fontSize: 12, opacity: 0.8 }}>
				Showing first 200 loci
			</div>
		</div>
	);
};

const th: React.CSSProperties = {
	textAlign: "left",
	padding: "10px 12px",
	borderTop: "1px solid #444",
	borderBottom: "1px solid #444",
	fontSize: 12,
	whiteSpace: "nowrap"
};

const td: React.CSSProperties = {
	padding: "10px 12px",
	borderTop: "1px solid #333",
	fontSize: 13,
	verticalAlign: "top"
};


