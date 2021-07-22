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

export TRAIN_CLUSTER=1
echo export TRAIN_CLUSTER

cd data/
python3 cifar10.py
