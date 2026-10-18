#!/usr/bin/env bash
# PipeGen Stage 4: TensorRT Engine Compiler Script
set -e

ONNX_MODEL="E:\hackathon project\stage3_model\yolov8n-pose.onnx"
ENGINE_FP16="E:\hackathon project\stage4_compiler\models\yolov8n-pose_fp16.engine"
ENGINE_INT8="E:\hackathon project\stage4_compiler\models\yolov8n-pose_int8.engine"
CALIB_CACHE="E:\hackathon project\stage4_compiler\models\yolov8n-pose_calib.cache"

echo "=========================================================="
echo " PipeGen TensorRT Compiler: YOLOv8n-Pose"
echo "=========================================================="

echo "[1/2] Building TensorRT FP16 Engine..."
trtexec --onnx="$ONNX_MODEL" \
        --saveEngine="$ENGINE_FP16" \
        --fp16 \
        --workspace=2048 \
        --avgRuns=100 \
        --dumpProfile \
        --exportTimes=trt_fp16_times.json

echo "[+] FP16 Engine successfully generated: $ENGINE_FP16"

echo "[2/2] Building TensorRT INT8 Quantized Engine..."
trtexec --onnx="$ONNX_MODEL" \
        --saveEngine="$ENGINE_INT8" \
        --int8 \
        --calib="$CALIB_CACHE" \
        --workspace=2048 \
        --avgRuns=100 \
        --dumpProfile \
        --exportTimes=trt_int8_times.json

echo "[+] INT8 Engine successfully generated: $ENGINE_INT8"
echo "[+] TensorRT Compilation Complete!"
