# MRF Demo - ISMRM 2022

This Demo is built to let others try to run the processing pipeline used in the abstract *Toward a 1-minute high-resolution brain exam - MR Fingerprinting with ML-synthesized contrasts and fast reconstruction* presented at the 2022 ISMRM Annual meeting in London (program number 53). You can find the abstract [here](https://submissions.mirasmart.com/ISMRM2022/itinerary/Files/PDFFiles/0053.html) (requires login).

This project is aimed at translating highly undrsampled MRF to a clinically feasible tool, by building a robust reconstruction pipeline that is portable between different compute systems (research lab, hospital, high performance computing cluster, collaborators, etc...). To achieve this these core objectives were set:

- The pipeline should run smoothly on multiple systems.
- The pipeline should be easy to upgrade when the sequence, the reconstruction method, or the synthesis method is changed.
- The pipeline should be able to provide an image to send back to the scanner within 5 min.
- The pipeline should be able to send a series of images to PACS within ~30min.
- The pipeline should run on hardware available in clinical settings (for now, this means an 11GB GPU).

This modular MRF processing pipeline includes 4 steps:

1.   Read raw scan data and metadata(In the demo this step is replaced with downloading a demo dataset)
2.   Reconstruct and get coil compression from calibration scan
3.   Reconstruct MRF (fast subspace basis reconstruction)
4.   Synthesize clinical contrasts

Each step will be demonstrated in a Google Colab session, and documentation on how to run the equivalent Docker/Singularity containers on your own machine will be explained in the final section of the Jupyter notebook.
