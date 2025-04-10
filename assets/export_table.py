import os
from datetime import datetime

import pandas as pd


class ResultsLogger:
    """
    A utility class to maintain a running table of results on disk,
    with automatic LaTeX table generation.
    """

    def __init__(self, filepath='results_log.csv', latex_filepath='results_table.tex'):
        self.filepath = filepath
        self.latex_filepath = latex_filepath

    def add_result(self, result_dict):
        """
        Add a new result row and update both CSV and LaTeX files.
        
        Args:
            result_dict (dict): Dictionary containing the results to add
                               Keys will be used as column names
        """
        # Add timestamp if not present
        if 'timestamp' not in result_dict:
            result_dict['timestamp'] = datetime.now()

        # Convert to DataFrame
        new_row = pd.DataFrame([result_dict])

        # If file exists, append to it
        if os.path.exists(self.filepath):
            try:
                df = pd.read_csv(self.filepath)
                new_columns = set(new_row.columns) - set(df.columns)
                if new_columns:
                    for col in new_columns:
                        df[col] = None
                df = pd.concat([df, new_row], ignore_index=True)
            except Exception as e:
                print(f"Error reading existing file: {e}")
                return
        else:
            df = new_row

        # Save to disk
        try:
            df.to_csv(self.filepath, index=False)
            self._update_latex_table(df)
            print(f"Successfully added new result to {self.filepath} and updated LaTeX table")
        except Exception as e:
            print(f"Error saving results: {e}")

    def _update_latex_table(self, df):
        """
        Convert the current results to a LaTeX table and save to disk.
        """
        # Format numeric columns to 3 decimal places
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].round(3)

        # Create LaTeX table
        latex_code = "\\begin{table}[h]\n\\centering\n\\begin{tabular}"

        # Create column specification
        n_cols = len(df.columns)
        col_spec = "{" + "c" * n_cols + "}"
        latex_code += col_spec + "\n\\toprule\n"

        # Add headers
        headers = " & ".join(df.columns) + " \\\\\n\\midrule\n"
        latex_code += headers

        # Add data rows
        for _, row in df.iterrows():
            row_str = " & ".join(str(val) for val in row) + " \\\\\n"
            latex_code += row_str

        # Close the table
        latex_code += "\\bottomrule\n\\end{tabular}\n"
        latex_code += "\\caption{Experimental Results}\n"
        latex_code += "\\label{tab:results}\n"
        latex_code += "\\end{table}"

        # Save LaTeX code
        with open(self.latex_filepath, 'w') as f:
            f.write(latex_code)

    def get_results(self):
        """
        Read and return all results as a pandas DataFrame
        """
        if os.path.exists(self.filepath):
            return pd.read_csv(self.filepath)
        return pd.DataFrame()

    def get_latex_code(self):
        """
        Return the current LaTeX table code as a string
        """
        if os.path.exists(self.latex_filepath):
            with open(self.latex_filepath) as f:
                return f.read()
        return ""


# Example usage:
if __name__ == "__main__":
    # Initialize logger
    logger = ResultsLogger('./assets/logged_results.csv', './assets/logged_results.tex')

    # Example of adding new results
    result1 = {
        'Model': 'CNN',
        'Accuracy': 0.856,
        'Loss': 0.234,
        'Params': 1.5e6
    }

    result2 = {
        'Model': 'Transformer',
        'Accuracy': 0.878,
        'Loss': 0.212,
        'Params': 2.8e6
    }

    logger.add_result(result1)
    logger.add_result(result2)

    # Print LaTeX code
    print("\nGenerated LaTeX table:")
    print(logger.get_latex_code())
