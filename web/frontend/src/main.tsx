import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";

const rootEl = document.getElementById("root");
if (!rootEl) {
	throw new Error("Root element #root not found");
}
const root = createRoot(rootEl);
root.render(<App />);


