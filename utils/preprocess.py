import pandas as pd

def ucl_dataset_prep(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess the UCL Household Energy dataset from the given file path.

    Steps:
    1. Reads the semicolon-delimited file, parsing Date & Time into a single datetime index.
    2. Drops rows with any missing values.
    3. Converts Global_active_power to float.
    4. Sets the datetime column as the DataFrame index.

    Parameters
    ----------
    filepath : str
        Path to the 'household_power_consumption.txt' file.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame ready for analysis.
    """
    df = pd.read_csv(
        filepath,
        sep=';',
        parse_dates={'datetime': ['Date', 'Time']},
        infer_datetime_format=True,
        na_values=['?'],
        low_memory=False
    )
    df = df.dropna()
    df['Global_active_power'] = df['Global_active_power'].astype(float)
    df = df.set_index('datetime')
    return df