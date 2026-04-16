for i in {0..9}
do
    echo "GPU $i:"
    export CUDA_VISIBLE_DEVICES=$i
    python -c "import torch; print(torch.cuda.is_available())"
done
