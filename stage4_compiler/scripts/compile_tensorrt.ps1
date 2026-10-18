# PipeGen Stage 4: TensorRT Engine Compiler Script (Windows / WSL)
$ErrorActionPreference = "Stop"

$OnnxModel = "E:\hackathon project\stage3_model\yolov8n-pose.onnx"
$EngineFP16 = "E:\hackathon project\stage4_compiler\models\yolov8n-pose_fp16.engine"
$EngineINT8 = "E:\hackathon project\stage4_compiler\models\yolov8n-pose_int8.engine"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " PipeGen TensorRT Compiler: YOLOv8n-Pose" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

Write-Host "[1/2] Building TensorRT FP16 Engine..." -ForegroundColor Yellow
trtexec --onnx="$OnnxModel" --saveEngine="$EngineFP16" --fp16 --workspace=2048 --avgRuns=100 --dumpProfile

Write-Host "[2/2] Building TensorRT INT8 Engine..." -ForegroundColor Yellow
trtexec --onnx="$OnnxModel" --saveEngine="$EngineINT8" --int8 --workspace=2048 --avgRuns=100 --dumpProfile

Write-Host "[+] TensorRT Compilation Complete!" -ForegroundColor Green
