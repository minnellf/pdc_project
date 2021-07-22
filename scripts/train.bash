#nvidia-smi
# source ~/.bashrc # not working, so the following is an alternative way
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/llavagno/miniconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/llavagno/miniconda/etc/profile.d/conda.sh" ]; then
        . "/home/llavagno/miniconda/etc/profile.d/conda.sh"
    else
        export PATH="/home/llavagno/miniconda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
echo conda initialized

conda activate pdc
echo conda env activated

LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-10.1/targets/x86_64-linux/lib/:/usr/lib64:./cuda/lib64:./cuda/include
echo LD_LIBRARY_PATH set

export LD_LIBRARY_PATH
echo export LD_LIBRARY_PATH

export TRAIN_CLUSTER=1
echo export TRAIN_CLUSTER

nvidia-smi

python py/train.py
