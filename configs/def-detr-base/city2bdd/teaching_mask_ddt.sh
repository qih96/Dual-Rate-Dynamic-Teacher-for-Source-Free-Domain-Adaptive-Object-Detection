N_GPUS=4
BATCH_SIZE=8
DATA_ROOT=./data
OUTPUT_DIR=./outputs/def-detr-base/city2bdd/teaching_mask_ddt

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 torchrun \
--rdzv_endpoint localhost:26508 \
--nproc_per_node=${N_GPUS} \
main.py \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 9 \
--dropout 0.0 \
--data_root ${DATA_ROOT} \
--source_dataset cityscapes \
--target_dataset bdd100k \
--batch_size ${BATCH_SIZE} \
--eval_batch_size ${BATCH_SIZE} \
--lr 2e-4 \
--lr_backbone 2e-5 \
--lr_linear_proj 2e-5 \
--alpha_ema 0.9999 \
--alpha_aema 0.9997 \
--epoch 5 \
--epoch_lr_drop 80 \
--mode teaching_mask \
--threshold 0.4 \
--threshold_s 0.5 \
--only_class_loss \
--output_dir ${OUTPUT_DIR} \
--resume ${OUTPUT_DIR}/../source_only/city2bdd_source_only_29_09.pth 2>&1 |tee ${OUTPUT_DIR}/logfile
