from typing import Callable

import numpy as np
import pandas as pd


def latex_table(csv_file: str, column_algnments: list[str], column_decimals: list[int | None], column_names: list[str | None], mean_std_columns: list[tuple[int, int]], units: list[str | None], value_mappers: list[Callable | None], is_longtable: bool = False) -> str:
    # 1. Load the DataFrame
    df = pd.read_csv(csv_file)

    # Convert mean_std mapping for easier lookup (maps source std col -> target mean col)
    mean_to_std = {mean: std for mean, std in mean_std_columns}
    std_to_skip = set(mean_to_std.values())

    formatted_rows = []
    num_cols = len(df.columns)

    # 2. Format the data row by row (clean values, no units here)
    for _, row in df.iterrows():
        formatted_row = []
        for i in range(num_cols):
            if i in std_to_skip:
                continue

            val = row.iloc[i]
            decimals = column_decimals[i]

            # Handle Mean + Std merging
            if i in mean_to_std:
                std_idx = mean_to_std[i]
                std_val = row.iloc[std_idx]
                std_decimals = column_decimals[std_idx]

                if decimals is not None and isinstance(val, (int, float)):
                    mean_str = f"{val:.{decimals}f}"
                else:
                    mean_str = str(val)

                if (
                    std_decimals is not None
                    and isinstance(std_val, (int, float))
                ):
                    std_str = f"{std_val:.{std_decimals}f}"
                else:
                    std_str = str(std_val)

                cell_str = f"{mean_str} ($\\pm$ {std_str})"
            else:
                # Format standard cell
                if isinstance(val, (int, float)) and np.isnan(val):
                    cell_str = ""
                elif decimals is not None and isinstance(val, (int, float)):
                    cell_str = f"{val:.{decimals}f}"
                else:
                    cell_str = str(val)

                if value_mappers[i] is not None:
                    cell_str = value_mappers[i](cell_str)

            formatted_row.append(cell_str)
        formatted_rows.append(formatted_row)

    # 3. Handle Header Names with Units and SD modifiers
    final_headers = []
    for i in range(num_cols):
        if i in std_to_skip:
            continue

        # Get base name
        header_text = (
            column_names[i] if column_names[i] is not None else df.columns[i]
        )

        # Append "(\pm SD)" if it's a merged column
        if i in mean_to_std:
            header_text = f"{header_text} ($\\pm$ SD)"

        # Append unit in parentheses if provided
        unit = units[i]
        if unit:
            header_text = f"{header_text} ({unit})"

        final_headers.append(f"\\textbf{{{header_text}}}")

    # 4. Calculate column widths for pretty text padding
    all_rows = [final_headers] + formatted_rows
    col_widths = [
        max(len(row[i]) for row in all_rows) for i in range(len(final_headers))
    ]

    # 5. Build the LaTeX String
    active_alignments = [
        column_algnments[i] for i in range(num_cols) if i not in std_to_skip
    ]
    alignment_str = "|".join(active_alignments)

    latex = []
    latex.append(f"\\begin{{longtable}}[ht]{{{alignment_str}}}" if is_longtable else f"\\begin{{table*}}[ht]")
    latex.append("    \\centering")
    if not is_longtable: latex.append(f"    \\begin{{tabular}}{{{alignment_str}}}")

    # Padded Header Row
    padded_headers = [
        final_headers[i].ljust(col_widths[i])
        for i in range(len(final_headers))
    ]
    latex.append("        " + " & ".join(padded_headers) + " \\\\")
    latex.append("        \\hline")

    # Padded Data Rows
    for row in formatted_rows:
        padded_cells = [
            row[i].ljust(col_widths[i]) for i in range(len(row))
        ]
        latex.append("        " + " & ".join(padded_cells) + " \\\\")

    if not is_longtable: latex.append("    \\end{tabular}")
    latex.append("    \\caption{TODO}")
    latex.append("    \\label{TODO}")
    latex.append("\\end{longtable}" if is_longtable else "\\end{table*}")

    return "\n".join(latex)