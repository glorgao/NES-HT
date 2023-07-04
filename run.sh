
# Baseline HT=0.0 
# BattleZone, Phoenix, Qbert, NameThisGame
# BeamRider, Breakout, Enduro, Pong, Seaquest, SpaceInvaders

# Population Size, 128 and 256 -> 128
# Learning Rate, 0.003 and 0.01 -> 0.01 

# srun --ntasks=1 --cpus-per-task=120 -q cpu-512 python main.py --n_jobs 128 --population_size 128 --env_name AtlantisNoFrameskip-v4 --seed 0 --learning_rate 0.01  --exp_name 'LR1e-2|HT=0.0|Pop=128|STD=0.02|Linear|LR=0.01|' --reward_process 'centered_rank' > /dev/null 2>&1 &
# srun --ntasks=1 --cpus-per-task=120 -q cpu-512 python main.py --n_jobs 128 --population_size 128 --env_name BattleZoneNoFrameskip-v4 --seed 0 --learning_rate 0.01  --exp_name 'LR1e-2|HT=0.0|Pop=128|STD=0.02|Linear|LR=0.01|' --reward_process 'centered_rank' > /dev/null 2>&1 &
# srun --ntasks=1 --cpus-per-task=120 -q cpu-512 python main.py --n_jobs 128 --population_size 128 --env_name BreakoutNoFrameskip-v4 --seed 0 --learning_rate 0.01  --exp_name 'LR1e-2|HT=0.0|Pop=128|STD=0.02|Linear|LR=0.01|' --reward_process 'centered_rank' > /dev/null 2>&1 &
# srun --ntasks=1 --cpus-per-task=120 -q cpu-512 python main.py --n_jobs 128 --population_size 128 --env_name PongNoFrameskip-v4 --seed 0 --learning_rate 0.01  --exp_name 'LR1e-2|HT=0.0|Pop=128|STD=0.02|Linear|LR=0.01|' --reward_process 'centered_rank' > /dev/null 2>&1 &
# srun --ntasks=1 --cpus-per-task=120 -q cpu-512 python main.py --n_jobs 128 --population_size 128 --env_name QbertNoFrameskip-v4 --seed 0 --learning_rate 0.01  --exp_name 'LR1e-2|HT=0.0|Pop=128|STD=0.02|Linear|LR=0.01|' --reward_process 'centered_rank' > /dev/null 2>&1 &
# srun --ntasks=1 --cpus-per-task=120 -q cpu-512 python main.py --n_jobs 128 --population_size 128 --env_name SkiingNoFrameskip-v4 --seed 0 --learning_rate 0.01  --exp_name 'LR1e-2|HT=0.0|Pop=128|STD=0.02|Linear|LR=0.01|' --reward_process 'centered_rank' > /dev/null 2>&1 &
# srun --ntasks=1 --cpus-per-task=120 -q cpu-512 python main.py --n_jobs 128 --population_size 128 --env_name VentureNoFrameskip-v4 --seed 0 --learning_rate 0.01  --exp_name 'LR1e-2|HT=0.0|Pop=128|STD=0.02|Linear|LR=0.01|' --reward_process 'centered_rank' > /dev/null 2>&1 &
# srun --ntasks=1 --cpus-per-task=120 -q cpu-512 python main.py --n_jobs 128 --population_size 128 --env_name ZaxxonNoFrameskip-v4 --seed 0 --learning_rate 0.01  --exp_name 'LR1e-2|HT=0.0|Pop=128|STD=0.02|Linear|LR=0.01|' --reward_process 'centered_rank' > /dev/null 2>&1 &

# # 
# srun --ntasks=1 --cpus-per-task=120 -q cpu-512 python main.py --n_jobs 128 --population_size 128 --env_name AtlantisNoFrameskip-v4 --seed 0 --learning_rate 0.01  --exp_name 'LR1e-2|HT=0.0|Pop=128|STD=0.02|MLP|' --reward_process 'centered_rank' > /dev/null 2>&1 &
# srun --ntasks=1 --cpus-per-task=120 -q cpu-512 python main.py --n_jobs 128 --population_size 128 --env_name BattleZoneNoFrameskip-v4 --seed 0 --learning_rate 0.01  --exp_name 'LR1e-2|HT=0.0|Pop=128|STD=0.02|MLP|' --reward_process 'centered_rank' > /dev/null 2>&1 &
# srun --ntasks=1 --cpus-per-task=120 -q cpu-512 python main.py --n_jobs 128 --population_size 128 --env_name BreakoutNoFrameskip-v4 --seed 0 --learning_rate 0.01  --exp_name 'LR1e-2|HT=0.0|Pop=128|STD=0.02|MLP|' --reward_process 'centered_rank' > /dev/null 2>&1 &
# srun --ntasks=1 --cpus-per-task=120 -q cpu-512 python main.py --n_jobs 128 --population_size 128 --env_name PongNoFrameskip-v4 --seed 0 --learning_rate 0.01  --exp_name 'LR1e-2|HT=0.0|Pop=128|STD=0.02|MLP|' --reward_process 'centered_rank' > /dev/null 2>&1 &
# srun --ntasks=1 --cpus-per-task=120 -q cpu-512 python main.py --n_jobs 128 --population_size 128 --env_name QbertNoFrameskip-v4 --seed 0 --learning_rate 0.01  --exp_name 'LR1e-2|HT=0.0|Pop=128|STD=0.02|MLP|' --reward_process 'centered_rank' > /dev/null 2>&1 &
# srun --ntasks=1 --cpus-per-task=120 -q cpu-512 python main.py --n_jobs 128 --population_size 128 --env_name SkiingNoFrameskip-v4 --seed 0 --learning_rate 0.01  --exp_name 'LR1e-2|HT=0.0|Pop=128|STD=0.02|MLP|' --reward_process 'centered_rank' > /dev/null 2>&1 &
# srun --ntasks=1 --cpus-per-task=120 -q cpu-512 python main.py --n_jobs 128 --population_size 128 --env_name VentureNoFrameskip-v4 --seed 0 --learning_rate 0.01  --exp_name 'LR1e-2|HT=0.0|Pop=128|STD=0.02|MLP|' --reward_process 'centered_rank' > /dev/null 2>&1 &
# srun --ntasks=1 --cpus-per-task=120 -q cpu-512 python main.py --n_jobs 128 --population_size 128 --env_name ZaxxonNoFrameskip-v4 --seed 0 --learning_rate 0.01  --exp_name 'LR1e-2|HT=0.0|Pop=128|STD=0.02|MLP|' --reward_process 'centered_rank' > /dev/null 2>&1 &

srun --ntasks=1 --cpus-per-task=120 -q cpu-512 python main.py --n_jobs 128 --population_size 128 --env_name AmidarNoFrameskip-v4 --seed 0 --learning_rate 0.01  --exp_name 'LR1e-2|HT=0.0|Pop=128|STD=0.02|MLP|' --reward_process 'centered_rank' > /dev/null 2>&1 &
srun --ntasks=1 --cpus-per-task=120 -q cpu-512 python main.py --n_jobs 128 --population_size 128 --env_name BowlingNoFrameskip-v4 --seed 0 --learning_rate 0.01  --exp_name 'LR1e-2|HT=0.0|Pop=128|STD=0.02|MLP|' --reward_process 'centered_rank' > /dev/null 2>&1 &
srun --ntasks=1 --cpus-per-task=120 -q cpu-512 python main.py --n_jobs 128 --population_size 128 --env_name BoxingNoFrameskip-v4 --seed 0 --learning_rate 0.01  --exp_name 'LR1e-2|HT=0.0|Pop=128|STD=0.02|MLP|' --reward_process 'centered_rank' > /dev/null 2>&1 &
srun --ntasks=1 --cpus-per-task=120 -q cpu-512 python main.py --n_jobs 128 --population_size 128 --env_name CentipedeNoFrameskip-v4 --seed 0 --learning_rate 0.01  --exp_name 'LR1e-2|HT=0.0|Pop=128|STD=0.02|MLP|' --reward_process 'centered_rank' > /dev/null 2>&1 &
srun --ntasks=1 --cpus-per-task=120 -q cpu-512 python main.py --n_jobs 128 --population_size 128 --env_name ChopperCommandNoFrameskip-v4 --seed 0 --learning_rate 0.01  --exp_name 'LR1e-2|HT=0.0|Pop=128|STD=0.02|MLP|' --reward_process 'centered_rank' > /dev/null 2>&1 &
srun --ntasks=1 --cpus-per-task=120 -q cpu-512 python main.py --n_jobs 128 --population_size 128 --env_name DemonAttackNoFrameskip-v4 --seed 0 --learning_rate 0.01  --exp_name 'LR1e-2|HT=0.0|Pop=128|STD=0.02|MLP|' --reward_process 'centered_rank' > /dev/null 2>&1 &
srun --ntasks=1 --cpus-per-task=120 -q cpu-512 python main.py --n_jobs 128 --population_size 128 --env_name NameThisGameNoFrameskip-v4 --seed 0 --learning_rate 0.01  --exp_name 'LR1e-2|HT=0.0|Pop=128|STD=0.02|MLP|' --reward_process 'centered_rank' > /dev/null 2>&1 &
srun --ntasks=1 --cpus-per-task=120 -q cpu-512 python main.py --n_jobs 128 --population_size 128 --env_name TimePilotNoFrameskip-v4 --seed 0 --learning_rate 0.01  --exp_name 'LR1e-2|HT=0.0|Pop=128|STD=0.02|MLP|' --reward_process 'centered_rank' > /dev/null 2>&1 &
ncc