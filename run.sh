# learning rate for hopper-v3: 0.1
# for slurm cluster
 
sigma=1.0
n_jobs=100 
for seed in 1..20
do
for red in 5 10 20 
do
srun --ntasks=1 --cpus-per-task=$n_jobs -q cpu-512 python main.py --n_jobs $n_jobs --env_name 'Hopper-v3' --seed $seed --learning_rate 0.1 --redundant $red --ht 0.0 --sigma $sigma --exp_name 'red20.0|ht0.0|sigma' --reward_process 'centered_rank' > /dev/null 2>&1 &
srun --ntasks=1 --cpus-per-task=$n_jobs -q cpu-512 python main.py --n_jobs $n_jobs --env_name 'Hopper-v3' --seed $seed --learning_rate 0.1 --redundant $red --ht 0.7 --sigma $sigma --exp_name 'red20.0|ht0.7|sigma' --reward_process 'centered_rank' > /dev/null 2>&1 &
srun --ntasks=1 --cpus-per-task=$n_jobs -q cpu-512 python main.py --n_jobs $n_jobs --env_name 'Hopper-v3' --seed $seed --learning_rate 0.1 --redundant $red --ht 0.9 --sigma $sigma --exp_name 'red20.0|ht0.9|sigma' --reward_process 'centered_rank' > /dev/null 2>&1 &
srun --ntasks=1 --cpus-per-task=$n_jobs -q cpu-512 python main.py --n_jobs $n_jobs --env_name 'Hopper-v3' --seed $seed --learning_rate 0.1 --redundant $red --ht 0.95 --sigma $sigma --exp_name 'red20.0|ht0.95|sigma' --reward_process 'centered_rank' > /dev/null 2>&1

srun --ntasks=1 --cpus-per-task=$n_jobs -q cpu-512 python main.py --n_jobs $n_jobs --env_name 'Walker2d-v3' --seed $seed --learning_rate 0.1 --redundant $red --ht 0.0 --sigma $sigma --exp_name 'red20.0|ht0.0|sigma' --reward_process 'centered_rank' > /dev/null 2>&1 &
srun --ntasks=1 --cpus-per-task=$n_jobs -q cpu-512 python main.py --n_jobs $n_jobs --env_name 'Walker2d-v3' --seed $seed --learning_rate 0.1 --redundant $red --ht 0.7 --sigma $sigma --exp_name 'red20.0|ht0.7|sigma' --reward_process 'centered_rank' > /dev/null 2>&1 &
srun --ntasks=1 --cpus-per-task=$n_jobs -q cpu-512 python main.py --n_jobs $n_jobs --env_name 'Walker2d-v3' --seed $seed --learning_rate 0.1 --redundant $red --ht 0.9 --sigma $sigma --exp_name 'red20.0|ht0.9|sigma' --reward_process 'centered_rank' > /dev/null 2>&1 &
srun --ntasks=1 --cpus-per-task=$n_jobs -q cpu-512 python main.py --n_jobs $n_jobs --env_name 'Walker2d-v3' --seed $seed --learning_rate 0.1 --redundant $red --ht 0.95 --sigma $sigma --exp_name 'red20.0|ht0.95|sigma' --reward_process 'centered_rank' > /dev/null 2>&1 

srun --ntasks=1 --cpus-per-task=$n_jobs -q cpu-512 python main.py --n_jobs $n_jobs --env_name 'HalfCheetah-v3' --seed $seed --learning_rate 0.1 --redundant $red --ht 0.0 --sigma $sigma --exp_name 'red20.0|ht0.0|sigma' --reward_process 'centered_rank' > /dev/null 2>&1 &
srun --ntasks=1 --cpus-per-task=$n_jobs -q cpu-512 python main.py --n_jobs $n_jobs --env_name 'HalfCheetah-v3' --seed $seed --learning_rate 0.1 --redundant $red --ht 0.7 --sigma $sigma --exp_name 'red20.0|ht0.7|sigma' --reward_process 'centered_rank' > /dev/null 2>&1 &
srun --ntasks=1 --cpus-per-task=$n_jobs -q cpu-512 python main.py --n_jobs $n_jobs --env_name 'HalfCheetah-v3' --seed $seed --learning_rate 0.1 --redundant $red --ht 0.9 --sigma $sigma --exp_name 'red20.0|ht0.9|sigma' --reward_process 'centered_rank' > /dev/null 2>&1 &
srun --ntasks=1 --cpus-per-task=$n_jobs -q cpu-512 python main.py --n_jobs $n_jobs --env_name 'HalfCheetah-v3' --seed $seed --learning_rate 0.1 --redundant $red --ht 0.95 --sigma $sigma --exp_name 'red20.0|ht0.95|sigma' --reward_process 'centered_rank' > /dev/null 2>&1 
done 
done
