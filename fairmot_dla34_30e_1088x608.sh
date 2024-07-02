#!/bin/bash

# configs/runtime.yml中save_dir改为: output/fairmot_dla34_30e_1088x608
# configs/datasets/mot.yml中EvalMOTDataset和TestMOTDataset改为: EvalDataset和TestDataset
# configs/yolov3/_base_/optimizer_40e.yml中base_lr改为: 1.25e-5

selected_gpus="0"
configs_file=configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml
output_dir=output/fairmot_dla34_30e_1088x608

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
    --use_vdl=True --eval > ${output_dir}/train.log 2>&1

if [ $? -ne 0 ]; then
    echo "Training failed"
    exit 1
else
    echo "Training finished, log saved in ${output_dir}/train.log"
fi

# evaluation
python tools/eval.py -c ${configs_file} \
    -o weights=${output_dir}/best_model.pdparams > ${output_dir}/eval.log 2>&1

if [ $? -ne 0 ]; then
    echo "Evaluation failed"
    exit 1
else
    echo "Evaluation finished, log saved in ${output_dir}/eval.log"
fi

# inference
python tools/infer.py -c ${configs_file} \
    --infer_img=demo/road554.png --output_dir=${output_dir} \
    -o weights=${output_dir}/best_model.pdparams > ${output_dir}/infer.log 2>&1

if [ $? -ne 0 ]; then
    echo "Inference failed"
    exit 1
else
    echo "Inference finished, log saved in ${output_dir}/infer.log"
fi