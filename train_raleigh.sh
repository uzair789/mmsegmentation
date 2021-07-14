export CUDA_VISIBLE_DEVICES='0'
CONFIG_FILE='configs/point_rend/pointrend_r50_512x1024_80k_raleigh.py'
GPU_NUM=2

python tools/train.py ${CONFIG_FILE}
#./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
