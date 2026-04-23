#!/usr/bin/bash

xmldir=FCCee/ALLEGRO/compact/ALLEGRO_o1_v03
xmlfileFullDet=ALLEGRO_o1_v03
xmlfileECal=ECalBarrel_thetamodulemerged

exec_cmd() {
    local cmd="$1"
    local log="$2"
    echo
    echo "Executing command $cmd"
    echo "Output will be recorded in file $log"
    echo
    # this will also show the output to screen
    # $cmd 2>&1 | tee $log
    # this redirects the output to file
    $cmd >  $log 2>&1
    retcode=${PIPESTATUS[0]}
    if [ $retcode -eq 0 ]; then
        echo -e "\e[1;42mDONE\e[0m"
        echo
    else
        echo -e "\e[1;41mFAILED\e[0m"
        exit
    fi
}

# Create noise maps for digitisation and topoclustering
# Some scripts need the sampling fractions, so this step should be run after the sampling fraction calculation
#
mkdir -p log
echo "Performing a dry run of the simulation (0 events) to obtain geometry parameters for the capacitance calculation..."
exec_cmd "ddsim --numberOfEvents 0 --enableGun --compactFile $K4GEO/$xmldir/$xmlfileFullDet.xml --runType run" "log/dryRunForNoise.log"

#echo "Extracting geometry and segmentation parameters for the capacitance calculation..."
#exec_cmd "python -u getECalBarrelNumbersForNoise.py" "log/getECalBarrelNumbersForNoise.log"

#echo "Creating the ecal barrel capacitance histograms..."
#exec_cmd "python -u create_capacitance_file_theta.py" "log/createCapacitanceECalBarrel.log"

#echo "Creating the ecal barrel noise histograms..."
#exec_cmd "python -u create_noise_file_chargePreAmp_theta.py" "log/createNoiseHistsECalBarrel.log"

#echo "Creating the ecal noise map..."
#exec_cmd "k4run noise_map_theta.py --detector $xmldir/$xmlfileFullDet.xml" "log/createNoiseMapEcal.log"

#echo "Creating the ecal+hcal noise map..."
#exec_cmd "k4run FCC-scripts/noise_map_theta.py --detector $xmldir/$xmlfileFullDet.xml --hcal" "log/createNoiseMapEcalHcal.log"

#echo "Noise maps created. To compare to a previous map to check if they are identical or not, you can use ../FCC-scripts/compareMaps.py"
