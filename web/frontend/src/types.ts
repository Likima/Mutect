export type Prediction = {
	sequence?: string;
	predicted_str?: boolean;
	str_probability?: number;
	repeat_motif?: string;
	repeat_count?: number;
	repeat_length?: number;
	repeat_purity?: number;
	repeat_coverage?: number;
	has_repeat?: boolean;
	// possible location fields
	chromosome?: string;
	position?: number;
	reference_name?: string;
	reference_start?: number;
	chrom?: string;
	start?: number;
	// system augmented
	_chromosome?: string | null;
	_position?: number | null;
	// other metadata
	[k: string]: unknown;
};

export type PredictionsPage = {
	total: number;
	page: number;
	page_size: number;
	items: Prediction[];
};

export type Summary = {
	total_sequences: number;
	predicted_strs: number;
	predicted_non_strs: number;
	average_str_probability: number;
	threshold: number;
	motif_statistics: {
		unique_motifs: number;
		most_common_motifs: Record<string, number>;
		motif_length_distribution: Record<string, number>;
	};
	top_20_str_predictions: Array<{
		sequence: string;
		probability: number;
		repeat_motif: string;
		repeat_count: number;
		repeat_length: number;
		chromosome: string;
		position: number;
	}>;
};

export type Locus = {
	locusId: string;
	chrom: string;
	start: number;
	end: number;
	repeatSequence: string;
	probability: number;
	readName: string;
	repeatMotif?: string;
	repeatCount?: number;
	repeatLength?: number;
};


