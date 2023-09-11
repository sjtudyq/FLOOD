python experiments.py --model=resnet \
	--dataset=cifar10 \
	--alg=openood \
	--lr=0.001 \
	--batch-size=64 \
	--epochs=200 \
	--n_parties=10 \
	--rho=0.9 \
	--comm_round=1 \
	--partition=noniid-#label1 \
	--beta=0.1\
	--device='cuda:1'\
	--datadir='./data/' \
	--logdir='./logs_logitnorm+pixmix_cifar10/' \
	--init_seed=0 \
	--config configs/datasets/cifar10/cifar10.yml \
		configs/preprocessors/pixmix_preprocessor.yml \
		configs/networks/resnet18_32x32.yml \
		configs/pipelines/train/train_logitnorm.yml \
		configs/postprocessors/msp.yml \
	--mark logitnorm+pixmix

python experiments.py --model=resnet \
	--dataset=cifar10 \
	--alg=openood \
	--lr=0.001 \
	--batch-size=64 \
	--epochs=200 \
	--n_parties=10 \
	--rho=0.9 \
	--comm_round=1 \
	--partition=noniid-#label2 \
	--beta=0.1\
	--device='cuda:1'\
	--datadir='./data/' \
	--logdir='./logs_logitnorm+pixmix_cifar10/' \
	--init_seed=0 \
	--config configs/datasets/cifar10/cifar10.yml \
		configs/preprocessors/pixmix_preprocessor.yml \
		configs/networks/resnet18_32x32.yml \
		configs/pipelines/train/train_logitnorm.yml \
		configs/postprocessors/msp.yml \
	--mark logitnorm+pixmix

python experiments.py --model=resnet \
	--dataset=cifar10 \
	--alg=openood \
	--lr=0.001 \
	--batch-size=64 \
	--epochs=200 \
	--n_parties=10 \
	--rho=0.9 \
	--comm_round=1 \
	--partition=noniid-#label3 \
	--beta=0.1\
	--device='cuda:1'\
	--datadir='./data/' \
	--logdir='./logs_logitnorm+pixmix_cifar10/' \
	--init_seed=0 \
	--config configs/datasets/cifar10/cifar10.yml \
		configs/preprocessors/pixmix_preprocessor.yml \
		configs/networks/resnet18_32x32.yml \
		configs/pipelines/train/train_logitnorm.yml \
		configs/postprocessors/msp.yml \
	--mark logitnorm+pixmix

python experiments.py --model=resnet \
	--dataset=cifar10 \
	--alg=openood \
	--lr=0.001 \
	--batch-size=64 \
	--epochs=200 \
	--n_parties=10 \
	--rho=0.9 \
	--comm_round=1 \
	--partition=noniid-labeldir \
	--beta=0.1\
	--device='cuda:1'\
	--datadir='./data/' \
	--logdir='./logs_logitnorm+pixmix_cifar10/' \
	--init_seed=0 \
	--config configs/datasets/cifar10/cifar10.yml \
		configs/preprocessors/pixmix_preprocessor.yml \
		configs/networks/resnet18_32x32.yml \
		configs/pipelines/train/train_logitnorm.yml \
		configs/postprocessors/msp.yml \
	--mark logitnorm+pixmix

python experiments.py --model=resnet \
	--dataset=cifar10 \
	--alg=openood \
	--lr=0.001 \
	--batch-size=64 \
	--epochs=200 \
	--n_parties=10 \
	--rho=0.9 \
	--comm_round=1 \
	--partition=noniid-labeldir \
	--beta=0.5\
	--device='cuda:1'\
	--datadir='./data/' \
	--logdir='./logs_logitnorm+pixmix_cifar10/' \
	--init_seed=0 \
	--config configs/datasets/cifar10/cifar10.yml \
		configs/preprocessors/pixmix_preprocessor.yml \
		configs/networks/resnet18_32x32.yml \
		configs/pipelines/train/train_logitnorm.yml \
		configs/postprocessors/msp.yml \
	--mark logitnorm+pixmix