cd models/YOLOv6
python3 tools/infer.py --yaml data/dataset.yaml --img-size 224 --weights runs/train/exp2/weights/best_ckpt.pt --source ../../data/images/
mkdir ../../out2
cp -r runs/inference/exp/ ../../out2/
# mkdir ../out2
# cp -r runs/inference/exp ../out2

cd ../..
pwd
echo stage 2 done