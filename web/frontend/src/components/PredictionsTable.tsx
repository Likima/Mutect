import React from "react";
import type { Prediction } from "../types";
import { fetchPredictions } from "../lib/api";

type Field = "pos" | "prob" | "motif" | "count" | "len" | "chrom";

const COLUMN_HELP: Record<Field, string> = {
	chrom: "Chromosome identifier (e.g., chr1, chr20).",
	pos: "Genomic start position for the sequence/record.",
	prob: "Model probability the sequence contains an STR (0–1).",
	motif: "Repeat motif sequence (e.g., TCCAT).",
	count: "Predicted repeat count (number of motif units).",
	len: "Predicted repeat length in bases (bp)."
};

function usePredictions() {
	const [items, setItems] = React.useState<Prediction[]>([]);
	const [total, setTotal] = React.useState(0);
	const [page, setPage] = React.useState(1);
	const [pageSize, setPageSize] = React.useState(50);
	const [minProb, setMinProb] = React.useState<number>(0.5);
	const [motif, setMotif] = React.useState("");
	const [sortBy, setSortBy] = React.useState<"pos" | "prob" | "motif" | "count" | "len" | "chrom">("prob");
	const [sortDir, setSortDir] = React.useState<"asc" | "desc">("desc");
	const [loading, setLoading] = React.useState(false);
	const [error, setError] = React.useState<string | null>(null);

	// Fetch the predictions
	React.useEffect(() => {
		const ctrl = new AbortController();
		setLoading(true);
		setError(null);
		fetchPredictions(
			{
				only_strs: true,
				min_prob: minProb,
				motif: motif || undefined,
				sort_by: sortBy,
				sort_dir: sortDir,
				page,
				page_size: pageSize
			},
			ctrl.signal
		)
			.then((r) => {
				setItems(r.items);
				setTotal(r.total);
			})
			.catch((e) => setError(String(e)))
			.finally(() => setLoading(false));
		return () => ctrl.abort();
	}, [page, pageSize, minProb, motif, sortBy, sortDir]);

	// Return the predictions table
	return {
		items,
		total,
		page,
		setPage,
		pageSize,
		setPageSize,
		minProb,
		setMinProb,
		motif,
		setMotif,
		sortBy,
		setSortBy,
		sortDir,
		setSortDir,
		loading,
		error
	};
}

// PredictionsTable component to display the predictions table
export const PredictionsTable: React.FC = () => {
	const s = usePredictions();
	const [expandedRows, setExpandedRows] = React.useState<Record<number, boolean>>({});

	// Toggle the sort order of the predictions table
	function toggleSort(field: Field) {
		if (s.sortBy === field) {
			s.setSortDir(s.sortDir === "asc" ? "desc" : "asc");
		} else {
			// Numeric fields default to desc, text to asc
			const defaultDir = field === "motif" || field === "chrom" ? "asc" : "desc";
			s.setSortBy(field);
			s.setSortDir(defaultDir);
		}
		// reset to first page when sorting changes
		s.setPage(1);
	}

	// Header label for the predictions table
	function headerLabel(field: Field, label: string) {
		const active = s.sortBy === field;
		const arrow = !active ? "↕" : s.sortDir === "asc" ? "▲" : "▼";
		return (
			<button
				onClick={() => toggleSort(field)}
				title={COLUMN_HELP[field]}
				style={{
					all: "unset",
					cursor: "pointer",
					display: "inline-flex",
					gap: 6,
					alignItems: "center"
				}}
				aria-label={`Sort by ${label}`}
			>
				<span>{label}</span>
				<span style={{ opacity: active ? 1 : 0.6 }}>{arrow}</span>
			</button>
		);
	}

	// Return the predictions table
	return (
		<div style={{ border: "1px solid #444", borderRadius: 8 }}>
			<div style={{ padding: 12, display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
				<label style={{ display: "flex", alignItems: "center", gap: 6 }}>
					<span style={{ fontSize: 12, opacity: 0.8 }}>Min probability</span>
					<input
						type="number"
						min={0}
						max={1}
						step={0.01}
						value={s.minProb}
						onChange={(e) => s.setMinProb(Number(e.target.value))}
						style={{ width: 90 }}
					/>
				</label>
				<label style={{ display: "flex", alignItems: "center", gap: 6 }}>
					<span style={{ fontSize: 12, opacity: 0.8 }}>Motif</span>
					<input
						type="text"
						placeholder="e.g. TCCAT"
						value={s.motif}
						onChange={(e) => s.setMotif(e.target.value)}
					/>
				</label>
				<span style={{ marginLeft: "auto", fontSize: 12, opacity: 0.8 }}>
					{Intl.NumberFormat().format(s.total)} results
				</span>
			</div>

			{ s.error ? (
				<div style={{ padding: 12, color: "tomato" }}>Failed to load predictions: {s.error}</div>
			) : s.loading ? (
				<div style={{ padding: 12 }}>Loading…</div>
			) : (
				<div style={{ overflow: "auto" }}>
					<table style={{ borderCollapse: "collapse", width: "100%" }}>
						<thead>
							<tr>
								<th style={th}>{headerLabel("chrom", "Chrom")}</th>
								<th style={th}>{headerLabel("pos", "Pos")}</th>
								<th style={th}>{headerLabel("prob", "Prob")}</th>
								<th style={th}>{headerLabel("motif", "Motif")}</th>
								<th style={th}>{headerLabel("count", "Count")}</th>
								<th style={th}>{headerLabel("len", "Len")}</th>
								<th style={th}>Seq (prefix)</th>
							</tr>
						</thead>
						<tbody>
							{s.items.map((p, idx) => {
								const isOpen = !!expandedRows[idx];
								const onToggle = () => setExpandedRows((prev) => ({ ...prev, [idx]: !prev[idx] }));
								return (
									<React.Fragment key={idx}>
										<tr>
											<td style={td}>{p._chromosome ?? p.chromosome ?? p.reference_name ?? "-"}</td>
											<td style={td}>{p._position ?? p.position ?? p.reference_start ?? "-"}</td>
											<td style={td}>{p.str_probability?.toFixed(3)}</td>
											<td style={td} title={String(p.repeat_motif ?? "")}>
												<code>{p.repeat_motif ?? ""}</code>
											</td>
											<td style={td}>{p.repeat_count ?? ""}</td>
											<td style={td}>{p.repeat_length ?? ""}</td>
											<td style={td}>
												<button
													onClick={onToggle}
													title="Click to view full sequence with highlighted repeats"
													style={{
														all: "unset",
														cursor: "pointer",
														fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
														background: "transparent",
														color: "inherit"
													}}
												>
													<code>
														{(p.sequence ?? "").slice(0, 32)}
														{(p.sequence ?? "").length > 32 ? "…" : ""}
													</code>
												</button>
											</td>
										</tr>
										{isOpen && (
											<tr>
												<td style={expandTd} colSpan={7}>
													<SequenceHighlight sequence={p.sequence ?? ""} motif={(p.repeat_motif ?? "").toString()} />
												</td>
											</tr>
										)}
									</React.Fragment>
								);
							})}
						</tbody>
					</table>
				</div>
			)}

			<div style={{ padding: 12, display: "flex", gap: 8, alignItems: "center" }}>
				<button onClick={() => s.setPage(Math.max(1, s.page - 1))} disabled={s.page <= 1}>
					Prev
				</button>
				<span style={{ fontSize: 12, opacity: 0.8 }}>Page {s.page}</span>
				<button
					onClick={() => s.setPage(s.page + 1)}
					disabled={s.page * s.pageSize >= s.total}
				>
					Next
				</button>
				<div style={{ marginLeft: "auto" }}>
					<label style={{ fontSize: 12 }}>
						Page size{" "}
						<select
							value={s.pageSize}
							onChange={(e) => s.setPageSize(Number(e.target.value))}
						>
							<option value={25}>25</option>
							<option value={50}>50</option>
							<option value={100}>100</option>
							<option value={200}>200</option>
						</select>
					</label>
				</div>
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

const expandTd: React.CSSProperties = {
	padding: 12,
	borderTop: "1px solid #333",
	background: "rgba(127,127,127,0.08)"
};

function SequenceHighlight({ sequence, motif }: { sequence: string; motif: string }) {
	if (!sequence) {
		return <div style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }}>(no sequence)</div>;
	}
	const cleanMotif = motif && motif !== "N/A" ? motif : "";
	if (!cleanMotif) {
		return (
			<div style={seqBox}>
				<code style={seqCode}>{sequence}</code>
			</div>
		);
	}
	const parts: Array<{ text: string; match: boolean }> = [];
	const motifLen = cleanMotif.length;
	let i = 0;
	while (i < sequence.length) {
		const next = sequence.indexOf(cleanMotif, i);
		if (next === -1) {
			parts.push({ text: sequence.slice(i), match: false });
			break;
		}
		if (next > i) {
			parts.push({ text: sequence.slice(i, next), match: false });
		}
		parts.push({ text: sequence.slice(next, next + motifLen), match: true });
		i = next + motifLen;
	}
	return (
		<div>
			<div style={{ marginBottom: 6, fontSize: 12, opacity: 0.8 }}>
				Highlighting motif <code>{cleanMotif}</code>
			</div>
			<div style={seqBox}>
				{parts.map((p, idx) =>
					p.match ? (
						<span key={idx} style={hl}>
							<code style={seqCode}>{p.text}</code>
						</span>
					) : (
						<span key={idx}>
							<code style={seqCode}>{p.text}</code>
						</span>
					)
				)}
			</div>
		</div>
	);
}

const seqBox: React.CSSProperties = {
	fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
	whiteSpace: "pre-wrap",
	wordBreak: "break-word",
	lineHeight: 1.6
};

const seqCode: React.CSSProperties = {
	fontFamily: "inherit"
};

const hl: React.CSSProperties = {
	background: "rgba(255, 208, 0, 0.35)",
	borderRadius: 3,
	padding: "0 1px"
};


