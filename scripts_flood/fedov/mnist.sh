for i in 0
do
	python experiments.py --model=lenet \
		--dataset=mnist \
		--alg=fedov \
		--lr=0.001 \
		--batch-size=64 \
		--epochs=200 \
		--n_parties=10 \
		--rho=0.9 \
		--comm_round=1 \
		--partition=noniid-#label1 \
		--beta=0.5\
		--device='cuda:0'\
		--datadir='./data/' \
		--logdir='./logs_mnist/' \
		--init_seed=$i \
		--config configs/datasets/mnist/mnist.yml \
			configs/preprocessors/base_preprocessor.yml \
			configs/networks/lenet.yml \
			configs/pipelines/train/baseline.yml

	python experiments.py --model=lenet \
		--dataset=mnist \
		--alg=fedov \
		--lr=0.001 \
		--batch-size=64 \
		--epochs=200 \
		--n_parties=10 \
		--rho=0.9 \
		--comm_round=1 \
		--partition=noniid-#label2 \
		--beta=0.5\
		--device='cuda:0'\
		--datadir='./data/' \
		--logdir='./logs_mnist/' \
		--init_seed=$i \
		--config configs/datasets/mnist/mnist.yml \
			configs/preprocessors/base_preprocessor.yml \
			configs/networks/lenet.yml \
			configs/pipelines/train/baseline.yml

	python experiments.py --model=lenet \
		--dataset=mnist \
		--alg=fedov \
		--lr=0.001 \
		--batch-size=64 \
		--epochs=200 \
		--n_parties=10 \
		--rho=0.9 \
		--comm_round=1 \
		--partition=noniid-#label3 \
		--beta=0.5\
		--device='cuda:0'\
		--datadir='./data/' \
		--logdir='./logs_mnist/' \
		--init_seed=$i \
		--config configs/datasets/mnist/mnist.yml \
			configs/preprocessors/base_preprocessor.yml \
			configs/networks/lenet.yml \
			configs/pipelines/train/baseline.yml

	python experiments.py --model=lenet \
		--dataset=mnist \
		--alg=fedov \
		--lr=0.001 \
		--batch-size=64 \
		--epochs=200 \
		--n_parties=10 \
		--rho=0.9 \
		--comm_round=1 \
		--partition=noniid-labeldir \
		--beta=0.1\
		--device='cuda:0'\
		--datadir='./data/' \
		--logdir='./logs_mnist/' \
		--init_seed=$i \
		--config configs/datasets/mnist/mnist.yml \
			configs/preprocessors/base_preprocessor.yml \
			configs/networks/lenet.yml \
			configs/pipelines/train/baseline.yml

	python experiments.py --model=lenet \
		--dataset=mnist \
		--alg=fedov \
		--lr=0.001 \
		--batch-size=64 \
		--epochs=200 \
		--n_parties=10 \
		--rho=0.9 \
		--comm_round=1 \
		--partition=noniid-labeldir \
		--beta=0.5\
		--device='cuda:0'\
		--datadir='./data/' \
		--logdir='./logs_mnist/' \
		--init_seed=$i \
		--config configs/datasets/mnist/mnist.yml \
			configs/preprocessors/base_preprocessor.yml \
			configs/networks/lenet.yml \
			configs/pipelines/train/baseline.yml
done

