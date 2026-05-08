# Architecture Diagrams

Place architecture diagram images here for inclusion in the report.

## Suggested Diagrams

| Filename | Description | Used In |
|---|---|---|
| `system_overview.png` | End-to-end pipeline: MIDI → Piano-Roll → Model → MIDI | Introduction / abstract |
| `task1_architecture.png` | LSTM Encoder–Decoder block diagram | Section 4.1 |
| `task2_architecture.png` | VAE with reparameterisation trick + genre embedding | Section 4.2 |
| `task3_architecture.png` | Transformer decoder layers + causal mask | Section 4.3 |
| `task4_rlhf.png` | RLHF feedback loop: Generator → Survey → Reward model → Policy gradient | Section 4.4 |
| `latent_interpolation.png` | Classical → Jazz latent space walk (8 steps) | Section 4.2 |

## How to Include in the Report

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\linewidth]{task1_architecture}
  \caption{LSTM Autoencoder architecture.}
  \label{fig:task1_arch}
\end{figure}
```

The `\graphicspath` directive in `final_report.tex` already points to this
directory, so no path prefix is needed—just the filename without extension.

## Tools for Creating Diagrams

- **draw.io / diagrams.net** — free, browser-based, exports PNG/PDF
- **TikZ / PGF** — native LaTeX; add `.tex` source files here and
  `\input` them directly
- **Lucidchart** — collaborative diagramming tool
- **PowerPoint / Keynote** — export slides as PNG at 300 dpi
