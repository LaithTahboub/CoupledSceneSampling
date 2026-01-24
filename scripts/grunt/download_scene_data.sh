id=$1
scene_name=$2
recon_no=$3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

s5cmd --no-sign-request cp s3://megascenes/images/$id/* ../test_final/$scene_name/images/
# s5cmd --no-sign-request cp s3://megascenes/metadata/subcat/$id/* ../scenes/$scene_name/metadata/
s5cmd --no-sign-request cp s3://megascenes/reconstruct/$id/colmap/$recon_no/* ../test_final/$scene_name/sparse/

# colmap image_undistorter --image_path ../test_final/$scene_name/images --input_path ../test_final/$scene_name/sparse --output_path ../test_final/$scene_name/undistort --output_type COLMAP
# mv ../test_final/$scene_name/undistort/images ../test_final/$scene_name/undistort/images_all
# python3 $SCRIPT_DIR/collect_images_into_one.py $scene_name
# python3 $SCRIPT_DIR/parse_images_bin.py $scene_name
# python3 $SCRIPT_DIR/convert_rgb.py $scene_name

# rm -rf ../test_final/$scene_name/undistort/images_all
# rm -rf ../test_final/$scene_name/undistort/images_copy
# rm -rf ../test_final/$scene_name/undistort/images_filtered
# rm -rf ../test_final/$scene_name/undistort/stereo
# rm -rf ../test_final/$scene_name/undistort/run-colmap-geometric.sh
# rm -rf ../test_final/$scene_name/undistort/run-colmap-photometric.sh
# #
# rm -rf ../test_final/$scene_name/images
# rm -rf ../test_final/$scene_name/sparse
# mv ../test_final/$scene_name/undistort/* ../test_final/$scene_name/
# rm -rf ../test_final/$scene_name/undistort

# python3 create_tsv.py $scene_name
# s5cmd --no-sign-request cp s3://megascenes/metadata/categories.json ../scenes/
