python experiments.py --model=lenet \
	--dataset=mnist \
	--alg=fedov \
	--lr=0.001 \
	--batch-size=64 \
	--epochs=20 \
	--n_parties=10 \
	--rho=0.9 \
	--comm_round=1 \
	--partition=noniid-#label2 \
	--beta=0.5\
	--device='cuda:0'\
	--datadir='./data/' \
	--logdir='./logs/' \
	--init_seed=0 \
	--config configs/datasets/mnist/mnist.yml \
		configs/preprocessors/base_preprocessor.yml \
		configs/networks/lenet.yml \
		configs/pipelines/train/baseline.yml
