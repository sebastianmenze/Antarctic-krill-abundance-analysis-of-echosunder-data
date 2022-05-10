# Antarctic-krill-abundance-analysis-of-echosunder-data

Python code to extract krill swarm backscatter from EK80 .raw files and estimate krill abundance base on transects, following the CCMLAR method

o read EK80 raw files, use this modification of pyecholab:

https://github.com/iambaim/pyEcholab/tree/fixes_for_crimac

to clean the echograms and detect swarm, use functions from echopy:

https://github.com/open-ocean-sounding/echopy

