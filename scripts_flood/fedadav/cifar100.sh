python experiments.py --model=resnet \
	--dataset=cifar100 \
	--alg=openood \
	--lr=0.001 \
	--batch-size=64 \
	--epochs=200 \
	--n_parties=100 \
	--rho=0.9 \
	--comm_round=1 \
	--partition=noniid-#label1 \
	--beta=0.1\
	--device='cuda:1'\
	--datadir='./data/' \
	--logdir='./logs_fedadav_cifar100/' \
	--init_seed=0 \
	--config configs/datasets/cifar100/cifar100.yml \
		configs/preprocessors/base_preprocessor.yml \
		configs/networks/fedadav_net.yml \
		configs/pipelines/train/train_fedadav.yml \
		configs/postprocessors/fedadav.yml \
	--mark fedadav

python experiments.py --model=resnet \
	--dataset=cifar100 \
	--alg=openood \
	--lr=0.001 \
	--batch-size=64 \
	--epochs=200 \
	--n_parties=100 \
	--rho=0.9 \
	--comm_round=1 \
	--partition=noniid-#label2 \
	--beta=0.1\
	--device='cuda:1'\
	--datadir='./data/' \
	--logdir='./logs_fedadav_cifar100/' \
	--init_seed=0 \
	--config configs/datasets/cifar100/cifar100.yml \
		configs/preprocessors/base_preprocessor.yml \
		configs/networks/fedadav_net.yml \
		configs/pipelines/train/train_fedadav.yml \
		configs/postprocessors/fedadav.yml \
	--mark fedadav

python experiments.py --model=resnet \
	--dataset=cifar100 \
	--alg=openood \
	--lr=0.001 \
	--batch-size=64 \
	--epochs=200 \
	--n_parties=100 \
	--rho=0.9 \
	--comm_round=1 \
	--partition=noniid-#label3 \
	--beta=0.1\
	--device='cuda:1'\
	--datadir='./data/' \
	--logdir='./logs_fedadav_cifar100/' \
	--init_seed=0 \
	--config configs/datasets/cifar100/cifar100.yml \
		configs/preprocessors/base_preprocessor.yml \
		configs/networks/fedadav_net.yml \
		configs/pipelines/train/train_fedadav.yml \
		configs/postprocessors/fedadav.yml \
	--mark fedadav

python experiments.py --model=resnet \
	--dataset=cifar100 \
	--alg=openood \
	--lr=0.001 \
	--batch-size=64 \
	--epochs=200 \
	--n_parties=100 \
	--rho=0.9 \
	--comm_round=1 \
	--partition=noniid-labeldir \
	--beta=0.1\
	--device='cuda:1'\
	--datadir='./data/' \
	--logdir='./logs_fedadav_cifar100/' \
	--init_seed=0 \
	--config configs/datasets/cifar100/cifar100.yml \
		configs/preprocessors/base_preprocessor.yml \
		configs/networks/fedadav_net.yml \
		configs/pipelines/train/train_fedadav.yml \
		configs/postprocessors/fedadav.yml \
	--mark fedadav

python experiments.py --model=resnet \
	--dataset=cifar100 \
	--alg=openood \
	--lr=0.001 \
	--batch-size=64 \
	--epochs=200 \
	--n_parties=100 \
	--rho=0.9 \
	--comm_round=1 \
	--partition=noniid-labeldir \
	--beta=0.5\
	--device='cuda:1'\
	--datadir='./data/' \
	--logdir='./logs_fedadav_cifar100/' \
	--init_seed=0 \
	--config configs/datasets/cifar100/cifar100.yml \
		configs/preprocessors/base_preprocessor.yml \
		configs/networks/fedadav_net.yml \
		configs/pipelines/train/train_fedadav.yml \
		configs/postprocessors/fedadav.yml \
	--mark fedadav