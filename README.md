# Classifying Special Quasars

This repository is home to a machine learning model built to classify Broad Absorption Line (BAL) and double-peaked quasars, aiding astrophycisists in the exploration of quasar features and their relation to cosmological phenomena like black hole and galaxy formation.


## Contents
- [Dataset](#dataset)
- [Feature Selection](#feature-selection)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Feature Importance](#feature-importance)
- [Replication (Setup & Usage)](#replication-setup--usage)


# Dataset
- **File:** 'data/Lyke2020.csv'
- **Source:** Lyke, B. W., Higley, A. N., McLane, J. N., Schurhammer, D. P., Myers, A. D., Ross, A. J., ... & Weaver, B. A. (2020). The Sloan Digital Sky Survey Quasar Catalog: Sixteenth Data Release. The Astrophysical Journal Supplement Series, 250(1), 8.
- The dataset contains hundreds of columns related to quasar identification, flux, magnitude, and other features.
- The target variable (BAL_PROB) indicates the BAL probability of each quasar.


# Preprocessing
- **Files:** 
-- preprocessing.ipynb
-- open_fits.py

Preprocessing complex cosmological data is no easy feat. However, the data has been previously processed by Lyke et. al. (2020), albiet for a different purpose.

The data was also originally supplied in a multidimensional .FITS file, which requires some work to convert to a much more usable .csv:
    
    import pandas
    from astropy.io import fits

    filename = 'DR16Q_v4.fits'
    with fits.open(filename) as hdul:
        # Accesses the first extension
        hdu = hdul[1] 
        data_dict = {}

        # Loops over each field in the binary table
        for name in hdu.columns.names:
            column_data = hdu.data[name]
        
            # Tests to see of the column is multi-dimensional
            # If true, flatten the column
            if column_data.ndim > 1:
                # Flattens each row's data and stores in the dictionary
                for i in range(column_data.shape[1]):
                    data_dict[f"{name}_{i}"] = column_data[:, i]
            else:
                # If data is not multi-dimensional, stores it as normal
                data_dict[name] = column_data

        df = pandas.DataFrame(data_dict)

        # Saves the DataFrame to a CSV file
        output_filename = 'output_table.csv'
        df.to_csv(output_filename, index=False)

A few thousand quasars in the dataset are, in fact, not quasars. These are removed rather simply:

    filtered_data = data[
        (data['IS_QSO_FINAL'] == 1)
    ]

Additionally, much of the data is unimportant for our purposes. Dozens upon dozens of columns are simply identifiers, and we don't need identifiers. Identifiers in this dataset all include the name 'DUPLICATE' in the column name. Thus, we remove all columns that contain that word.

    filtered_data = filtered_data.loc[:, ~filtered_data.columns.str.contains('DUPLICATE')]

For the purpose of creating a solid model, we want to remove any quasars that the scientists who created the original dataset were not absolutely certain were or were not BAL quasars.

    filtered_data = filtered_data[
        (data['BAL_PROB'] == 1) | (data['BAL_PROB'] == 0)
    ]

Lastly, we want to remove any dramatic outliers. These are inevitable in most huge datasets, and can introduce bias into models.

    cols = ['PSFFLUX_0', 'PSFFLUX_1', 'PSFFLUX_2', 'PSFFLUX_3', 'PSFFLUX_4',
            'FUV', 'NUV',
            'YFLUX', 'JFLUX', 'HFLUX', 'KFLUX',
            'W1_FLUX', 'W2_FLUX',
            'FIRST_FLUX',
            'XMM_SOFT_FLUX', 'XMM_HARD_FLUX',
            'GAIA_PARALLAX',
            'GAIA_G_FLUX_SNR', 'GAIA_BP_FLUX_SNR', 'GAIA_RP_FLUX_SNR']

    for col in cols:
        Q1 = filtered_data[col].quantile(0.25)
        Q3 = filtered_data[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        filtered_data[col] = numpy.where((filtered_data[col] < lower_bound) | (filtered_data[col] > upper_bound), numpy.nan, filtered_data[col])

# Feature Selection


# Model Training

# Evaluation

# Feature Importance

# Replication (Setup & Usage)