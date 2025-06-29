#!/bin/bash

# Activate conda environment
# Initialize conda
eval "$(conda shell.bash hook)"
conda activate mukh-dev

# Function to display usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -f    Run face detection tests"
    echo "  -r    Run reenactment tests"
    echo "  -d    Run deepfake detection tests"
    echo "  -p    Run deepfake detection pipeline tests"
    echo "  -a    Run all tests"
    exit 1
}

# Check for arguments
if [ $# -eq 0 ]; then
    usage
fi

# Clean output directory
echo "\nCleaning output directory..."
rm -rf output
echo "Output directory cleaned"
echo "--------------------------------"

# Parse options
while getopts "frdpa" opt; do
    case ${opt} in
        f )
            echo "Testing Face Detection:"
            echo "--------------------------------"
            echo ""
            echo "Testing blazeface"
            python -m examples.face_detection.basic_detection_single_image --detection_model blazeface
            echo "Completed testing blazeface"
            echo ""
            echo "Testing ultralight"
            python -m examples.face_detection.basic_detection_single_image --detection_model ultralight
            echo "Completed testing ultralight"
            echo ""
            echo "Testing mediapipe"
            python -m examples.face_detection.basic_detection_single_image --detection_model mediapipe
            echo "Completed testing mediapipe"
            ;;
        r )
            echo "Testing Reenactment:"
            echo "--------------------------------"
            echo ""
            echo "Testing tps"
            python -m examples.reenactment.basic_reenactment --reenactor_model tps
            echo "Completed testing tps"
            ;;
        d )
            echo "Testing Deepfake Detection:"
            echo "--------------------------------"
            echo ""
            echo "Testing resnet_inception"
            python -m examples.deepfake_detection.detection --detection_model resnet_inception
            echo "Completed testing resnet_inception"
            echo ""
            echo "Testing efficientnet"
            python -m examples.deepfake_detection.detection --detection_model efficientnet
            echo "Completed testing efficientnet"
            ;;
        p )
            echo "Testing Deepfake Detection Pipeline:"
            echo "--------------------------------"
            python -m examples.pipelines.deepfake_detection
            echo "Completed testing deepfake detection pipeline"
            ;;
        a )
            echo "Testing Face Detection:"
            echo "--------------------------------"
            echo ""
            echo "Testing blazeface"
            python -m examples.face_detection.basic_detection_single_image --detection_model blazeface
            echo "Completed testing blazeface"
            echo ""
            echo "Testing ultralight"
            python -m examples.face_detection.basic_detection_single_image --detection_model ultralight
            echo "Completed testing ultralight"
            echo ""
            echo "Testing mediapipe"
            python -m examples.face_detection.basic_detection_single_image --detection_model mediapipe
            echo "Completed testing mediapipe"
            echo "--------------------------------"

            echo "Testing Reenactment:"
            echo "--------------------------------"
            echo ""
            echo "Testing tps"
            python -m examples.reenactment.basic_reenactment --reenactor_model tps
            echo "Completed testing reenactment"
            echo "--------------------------------"

            echo "Testing Deepfake Detection:"
            echo "--------------------------------"
            echo ""
            echo "Testing resnet_inception"
            python -m examples.deepfake_detection.detection --detection_model resnet_inception
            echo "Completed testing resnet_inception"
            echo ""     
            echo "Testing efficientnet"
            python -m examples.deepfake_detection.detection --detection_model efficientnet
            echo "Completed testing efficientnet"
            echo "--------------------------------"

            echo "Testing Deepfake Detection Pipeline:"
            echo "--------------------------------"
            echo "" 
            echo "Testing deepfake detection pipeline"
            python -m examples.pipelines.deepfake_detection
            echo "Completed testing deepfake detection pipeline"
            echo "--------------------------------"
            ;;
        * )
            usage
            ;;
    esac
done
