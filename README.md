# Drifters analysis for the OTC25 paper

## Structure

- [data](data) contains the raw, pre-processed, analysis, etc... data files,
- [plots](plots) contains the plots produced,
- [src](src) contains the code used.

## Running the analysis again

- [src/requirements.txt](src/requirements.txt) contains the Python requirements to install first,
- then each step of the analysis has its main script under [src](src) that can be run from the command line,
- the dependencies for each main script lives in their corresponding sub-directory under [src](src), with shared utility functions in their own sub-directory,
- the top main srcipt [src](src) can be used to run all the steps at once.
