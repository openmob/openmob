![Open Urban Mobility Toolkit (OpenMob)](/figures/title.png)
# Open Urban Mobility Toolkit (OpenMob)

This project contains several preprocessing tools built by python:
* Basic function to detect stay points given GPS trajectory datasets.
  ![Stay Points Visualization](/figures/stay_points_vis.png)

* Building life pattern tree from raw GPS trajectory data.
* Preprocessing tools are used for the generation model of the "pseudo life pattern." 
* Essential preprocessing tools for converting raw GPS data to grid-based data format.
* Pseudo life pattern to the spatial sequence of trips.
* Map-matching as post-processing of GPS trajectory.

## Project structure
The following is the structure of this project.
```
openmob
├── datasets
│   ├── dataset_tsmc2014
│   ├── gis
│   ├── open_travel_mode_data
│   └── ...
├── scripts
│   ├── 1.py
│   ├── 2.py
│   ├── 3.py
│   └── ...
└── functions
    ├── stay_point_detection
    ├── life_pattern_processing
    ├── demographic_processing
    └── ...
```

## Datasets
* NYC and Tokyo Check-in Dataset. For more details, please refer to [Foursquare Dataset](https://sites.google.com/site/yangdingqi/home/foursquare-dataset).
* Open Travel Mode Data. For more details, please refer to this [paper](https://arxiv.org/pdf/2109.08527.pdf).
* GIS data. Collected from 

## Usage
This project is built from Python=3.8.4.\
Please refer to the [requirements.txt](requirements.txt) for the necessary python library.

## Authors and acknowledgment
We will add this part later shortly. <br />
Haoran Zhang: ... <br />
Dou Huang: <br />

## License
OpenMOB is released under the MIT license. Please refer to [LICENSES](LICENSE) for the careful check.

## Projects in OpenLab
* **[OpenMob](https://github.com/openmob/openmob)**: This project is currently under development. We will continuously update this project.
* **[OpenPV](https://github.com/OpenSolarPV/OpenPV)**: an open source toolbox for solar photovoltaics semantic segmentation.