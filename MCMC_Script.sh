#!/bin/sh

i=0
while [ $i -lt 7 ];
do
python example_bliss_transit_fitting.py -f ~/Research/GJ1214/data/group"$i"_gsc.joblib.save -p gj1214b_planet_params.json
i=$i+1
done
