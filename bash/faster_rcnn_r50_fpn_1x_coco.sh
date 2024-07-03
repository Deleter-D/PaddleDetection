#!/bin/bash

# configs/runtime.yml中save_dir改为: output/faster_rcnn_r50_fpn_1x_coco
# configs/datasets/coco_detection.yml中dataset_dir改为: dataset/coco/cocomini
# configs/faster_rcnn/_base_/optimizer_1x.yml中learning_rate改为: 1.25e-3

selected_gpus="0"
configs_file=configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml
output_dir=output/faster_rcnn_r50_fpn_1x_coco

if [ ! -d ${output_dir} ]; then
    mkdir -p "$output_dir"
fi

nv_gpu=$(lspci | grep -i nvidia)
hygon_gpu=$(lspci | grep -i display | grep -i haiguang)

if [ -n "$nv_gpu" ]; then
    echo "Nvidia GPU is detected"
    export CUDA_VISIBLE_DEVICES=$selected_gpus
elif [ -n "$hygon_gpu" ]; then
    echo "Hygon GPU is detected"
    export HIP_VISIBLE_DEVICES=$selected_gpus
else
    echo "No GPU is detected"
    exit 1
fi

# training standalone
python tools/train.py -c ${configs_file} \
    --use_vdl=True --eval >"${output_dir}/train.log" 2>&1

if [ $? -ne 0 ]; then
    echo "Training failed"
    exit 1
else
    echo "Training finished, log saved in ${output_dir}/train.log"
fi

# evaluation
python tools/eval.py -c ${configs_file} \
    -o weights=${output_dir}/best_model.pdparams >"${output_dir}/eval.log" 2>&1

if [ $? -ne 0 ]; then
    echo "Evaluation failed"
    exit 1
else
    echo "Evaluation finished, log saved in ${output_dir}/eval.log"
fi

# inference
python tools/infer.py -c ${configs_file} \
    --infer_img=dataset/coco/cocomini/val2017/000000000885.jpg --output_dir=${output_dir} \
    -o weights=${output_dir}/best_model.pdparams >"${output_dir}/infer.log" 2>&1

if [ $? -ne 0 ]; then
    echo "Inference failed"
    exit 1
else
    echo "Inference finished, log saved in ${output_dir}/infer.log"
fi
