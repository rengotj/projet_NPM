This project has been realised by Juliette Rengot, from February, 2019 to March, 2019, within the scope of MVA master (Nuage de points et modélisation 3D). François Goulette has supervised this work.

This project proposes an implementation of the Interest Point Detection (IPD) method described in "A robust 3D interest points detector based on Harris operator" (Eurographics workshop on 3D object retrieval) by I. Pratikakis, M. Spagnuolo, T. Theoharis and R. Veltkamp (2010).
The details of this method and a presentation of the obtained results are available in "ProjectReport_RENGOT_Juliette.pdf". "ProjectSlides_RENGOT_Juliette.pdf" contains a set of slides about this project.

The proposed method adapts the well-known Harris corner detector for images to point clouds. A complete description of the basic Harris detecteur and an implementation is available in the folder "Harris_IDP_on_images".

The code is contained into 4 files :
  3D_harris.py : the main function that contains the implementation of Harris IPD on 3D models
  neighbourhoods.py : a module defining the neighbourhoods (shperical, K-nearest, K-ring, adaptive K-ring)
  repeatability.py : a module computing the repeatability for the models stored in "data/data_to_compute_repeatability"
  transformation.py : a module to apply transformations (scaling, noise addition, rotation, change of resolution...) to an original model. Some of the obtained results can be seen in the folder "data/transformed".

The folder "utils" contains usefull fonction to read and vrite points cloud data.

The folder ''data'' contains some 3D model to test the model. It also has a folder "results" whith the obtained results for several experimentations.
