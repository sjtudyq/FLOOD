for i in 0
do
	python experiments.py --model=resnet \
		--dataset=cifar100 \
		--alg=fedov \
		--lr=0.001 \
		--batch-size=64 \
		--epochs=200 \
		--n_parties=100 \
		--rho=0.9 \
		--comm_round=1 \
		--partition=noniid-#label1 \
		--beta=0.5\
		--device='cuda:2'\
		--datadir='./data/' \
		--logdir='./logs_cifar100/' \
		--init_seed=$i \
		--config configs/datasets/cifar100/cifar100.yml \
			configs/preprocessors/base_preprocessor.yml \
			configs/networks/resnet18_32x32.yml \
			configs/pipelines/train/baseline.yml

	python experiments.py --model=resnet \
		--dataset=cifar100 \
		--alg=fedov \
		--lr=0.001 \
		--batch-size=64 \
		--epochs=200 \
		--n_parties=100 \
		--rho=0.9 \
		--comm_round=1 \
		--partition=noniid-#label2 \
		--beta=0.5\
		--device='cuda:2'\
		--datadir='./data/' \
		--logdir='./logs_cifar100/' \
		--init_seed=$i \
		--config configs/datasets/cifar100/cifar100.yml \
			configs/preprocessors/base_preprocessor.yml \
			configs/networks/resnet18_32x32.yml \
			configs/pipelines/train/baseline.yml

	python experiments.py --model=resnet \
		--dataset=cifar100 \
		--alg=fedov \
		--lr=0.001 \
		--batch-size=64 \
		--epochs=200 \
		--n_parties=100 \
		--rho=0.9 \
		--comm_round=1 \
		--partition=noniid-#label3 \
		--beta=0.5\
		--device='cuda:2'\
		--datadir='./data/' \
		--logdir='./logs_cifar100/' \
		--init_seed=$i \
		--config configs/datasets/cifar100/cifar100.yml \
			configs/preprocessors/base_preprocessor.yml \
			configs/networks/resnet18_32x32.yml \
			configs/pipelines/train/baseline.yml

	python experiments.py --model=resnet \
		--dataset=cifar100 \
		--alg=fedov \
		--lr=0.001 \
		--batch-size=64 \
		--epochs=200 \
		--n_parties=100 \
		--rho=0.9 \
		--comm_round=1 \
		--partition=noniid-labeldir \
		--beta=0.1\
		--device='cuda:2'\
		--datadir='./data/' \
		--logdir='./logs_cifar100/' \
		--init_seed=$i \
		--config configs/datasets/cifar100/cifar100.yml \
			configs/preprocessors/base_preprocessor.yml \
			configs/networks/resnet18_32x32.yml \
			configs/pipelines/train/baseline.yml

	python experiments.py --model=resnet \
		--dataset=cifar100 \
		--alg=fedov \
		--lr=0.001 \
		--batch-size=64 \
		--epochs=200 \
		--n_parties=100 \
		--rho=0.9 \
		--comm_round=1 \
		--partition=noniid-labeldir \
		--beta=0.5\
		--device='cuda:2'\
		--datadir='./data/' \
		--logdir='./logs_cifar100/' \
		--init_seed=$i \
		--config configs/datasets/cifar100/cifar100.yml \
			configs/preprocessors/base_preprocessor.yml \
			configs/networks/resnet18_32x32.yml \
			configs/pipelines/train/baseline.yml

done

