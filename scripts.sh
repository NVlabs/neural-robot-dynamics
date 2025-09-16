##################################################################################3
# Pendulum
# [Generate datasets]:
# cd generate
# python generate_dataset.py --env-name Pendulum --mode transition --num-transitions 1000000 --dataset-name train_transition.hdf5 --num-envs 1024 --seed 10
# python generate_dataset.py --env-name Pendulum --mode transition --num-transitions 1000000 --dataset-name valid_transition.hdf5 --num-envs 1024 --seed 20
# python generate_dataset.py --env-name Pendulum --mode trajectory --num-transitions 1000000 --dataset-name passive_trajectory.hdf5 --num-envs 1024 --seed 20 --passive
# [Train]
# python train.py --cfg ./cfg/Pendulum/naive_sl.yaml --logdir ./trained_models/Pendulum/train-transition/ --eval-interval 1