import React from "react";
import { SummaryCards } from "./components/SummaryCards";
import { PredictionsTable } from "./components/PredictionsTable";
import { LociList } from "./components/LociList";

export const App: React.FC = () => {
	return (
		<div style={{ display: "flex", flexDirection: "column", width: "100%" }}>
			<header style={{ padding: "16px 24px", borderBottom: "1px solid #444" }}>
				<h1 style={{ margin: 0, fontSize: 20 }}>Mutect</h1>
				<p style={{ margin: "6px 0 0 0", opacity: 0.8 }}>
					Predicting and Visualizing Short Tandem Repeats (STRs) in Long Read Sequencing Data
				</p>
			</header>

			<main style={{ display: "grid", gap: 16, padding: 16 }}>
				<SummaryCards />
				<section>
					<h2 style={{ fontSize: 16, margin: "0 0 8px 0" }}>Predicted STRs</h2>
					<PredictionsTable />
				</section>
				<section>
					<h2 style={{ fontSize: 16, margin: "0 0 8px 0" }}>Loci</h2>
					<LociList />
				</section>
			</main>
		</div>
	);
};


