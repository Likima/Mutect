import type { Locus, PredictionsPage, Summary } from "../types";

type SortBy = "pos" | "prob" | "motif" | "count" | "len" | "chrom";
type SortDir = "asc" | "desc";

export type PredictionsRequest = {
	only_strs?: boolean;
	min_prob?: number;
	motif?: string;
	chromosome?: string;
	has_repeat?: boolean;
	page?: number;
	page_size?: number;
	sort_by?: SortBy;
	sort_dir?: SortDir;
};

async function fetchJson<T>(input: string, signal?: AbortSignal): Promise<T> {
	const res = await fetch(input, { signal, headers: { Accept: "application/json" } });
	if (!res.ok) {
		let details = "";
		try {
			const text = await res.text();
			details = text ? ` — ${text}` : "";
		} catch {
			// ignore
		}
		throw new Error(`Request failed (${res.status} ${res.statusText})${details}`);
	}
	return res.json() as Promise<T>;
}

export async function fetchSummary(signal?: AbortSignal): Promise<Summary> {
	return fetchJson<Summary>("/api/summary", signal);
}

export async function fetchPredictions(params: PredictionsRequest, signal?: AbortSignal): Promise<PredictionsPage> {
	const qs = new URLSearchParams();
	if (params.only_strs !== undefined) qs.set("only_strs", String(params.only_strs));
	if (params.min_prob !== undefined) qs.set("min_prob", String(params.min_prob));
	if (params.motif) qs.set("motif", params.motif);
	if (params.chromosome) qs.set("chromosome", params.chromosome);
	if (params.has_repeat !== undefined) qs.set("has_repeat", String(params.has_repeat));
	if (params.page !== undefined) qs.set("page", String(params.page));
	if (params.page_size !== undefined) qs.set("page_size", String(params.page_size));
	if (params.sort_by) qs.set("sort_by", params.sort_by);
	if (params.sort_dir) qs.set("sort_dir", params.sort_dir);
	const url = `/api/predictions?${qs.toString()}`;
	return fetchJson<PredictionsPage>(url, signal);
}

export async function fetchLoci(signal?: AbortSignal): Promise<{ count: number; loci: Locus[] }> {
	return fetchJson<{ count: number; loci: Locus[] }>("/api/loci", signal);
}


