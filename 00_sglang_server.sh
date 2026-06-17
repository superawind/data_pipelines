# nohup python -m sglang.launch_server --port=8027 --tp-size=2 --trust-remote-code --host 10.16.80.9  --mem-fraction-static 0.88 --model-path /opt/users/models/Qwen3-Embedding-0.6B --attention-backend fa3 --is-embedding >> sglang.log &  # for embedding 
# nohup python -m sglang.launch_server --port=8027 --tp-size=4 --trust-remote-code --host 10.16.80.9  --mem-fraction-static 0.83 --model-path  /opt/users/models/Qwen3-30B-A3B-Instruct-2507/Qwen/Qwen3-30B-A3B-Instruct-2507 --attention-backend fa3 >> sglang.log &  # for cls

export CUDA_VISIBLE_DEVICES=0,1,2,3
export SGL_ENABLE_JIT_DEEPGEMM=false
source /opt/rh/gcc-toolset-9/enable
nohup python -m sglang.launch_server --model-path  /opt/users/models/Qwen3-235B-A22B-Instruct-2507-FP8  --port=8057 --tp-size=4 --trust-remote-code --host 10.16.80.154 --mem-fraction-static 0.83 --enable-dp-attention --dp-size 4 >> 154_2507-fp8.log &  # for response 