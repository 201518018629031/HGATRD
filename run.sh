GPU_ID=$1
DATASET=$2
EPOCHS=${3:-30}
JOINT_DIM=${4:-300}
GAT_HIDDEN_DIM=${5:-32}
ELAPSED_TIME=${6:-3000}
TWEETS_COUNT=${7:-500}
DROPOUT=${8:-"0.5"}
ALPHA=${9:-"0.3"}
echo "gpu_id:${GPU_ID} dataset:${DATASET} epochs:${EPOCHS} joint_dim:${JOINT_DIM} gat_hidden_dim:${GAT_HIDDEN_DIM} elapsed_time:${ELAPSED_TIME} tweets_count:${TWEETS_COUNT} dropout:${DROPOUT} alpha:${ALPHA}"
#LOG="logs/${DATASET}_f${FOLD_IDX}_100_w4_s_uatt.log"
#exec &> >(tee -a "$LOG")
#echo Logging output to "$LOG"

CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py --dataset ${DATASET} --epochs ${EPOCHS} --joint_dim ${JOINT_DIM} --gat_hidden_dim ${GAT_HIDDEN_DIM} --elapsed_time ${ELAPSED_TIME} --tweets_count ${TWEETS_COUNT} --filename result/${DATASET}_et${ELAPSED_TIME}_tc${TWEETS_COUNT}.txt --dropout ${DROPOUT} --alpha ${ALPHA}
